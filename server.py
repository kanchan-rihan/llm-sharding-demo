# server.py
import os
import time
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# =========================
# CONFIGURATION (Restaurant setup)
# =========================
# Think of SHARD_ROLE as defining the role of each "kitchen station":
# - "a": a prep chef who chops ingredients (Shard A)
# - "b": a line cook who finishes the dish and plates it (Shard B)
# - "coordinator": the head chef taking orders from customers and orchestrating A and B
MODEL_ID = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
SHARD_ROLE = os.environ.get("SHARD_ROLE", "coordinator").lower()
SPLIT_AT = int(os.environ.get("SPLIT_AT", "1"))
SHARD_A_SERVICE = os.environ.get("SHARD_A_SERVICE", "llm-shard-a")
SHARD_B_SERVICE = os.environ.get("SHARD_B_SERVICE", "llm-shard-b")
SHARD_PORT = int(os.environ.get("SHARD_PORT", "5000"))

print("Starting server role:", SHARD_ROLE, "model:", MODEL_ID)

# =========================
# DEVICE SELECTION
# =========================
# In this demo we use CPU, since Minikube containers won’t expose GPU (no ovens here).
device = torch.device("cpu")

# =========================
# LOAD MODEL + TOKENIZER
# =========================
# The tokenizer = waiter who takes the text order and converts it into IDs (ingredients).
# The model = recipe book + kitchen staff who know how to cook.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

# =========================
# MODEL SPLITTING (Kitchen workflow division)
# =========================
# Large LLMs don’t fit into one GPU (kitchen too small).
# We split model into two shards:
#   Shard A = embedding + first few blocks (prep chef)
#   Shard B = remaining blocks + output head (line cook who finishes the dish)
def split_gpt2_model(model, split_at=1):
    gpt2 = model.transformer
    lm_head = model.lm_head

    # Base components
    wte = gpt2.wte  # word embeddings
    wpe = gpt2.wpe  # position embeddings
    drop = gpt2.drop
    blocks = gpt2.h  # transformer blocks
    ln_f = gpt2.ln_f

    # Split blocks into two groups (two kitchen stations)
    blocks_a = nn.ModuleList([blocks[i] for i in range(split_at)])
    blocks_b = nn.ModuleList([blocks[i] for i in range(split_at, len(blocks))])

    # --- Shard A ---
    # Chef who preps the ingredients: token embeddings + early layers
    class ShardA(nn.Module):
        def __init__(self, wte, wpe, drop, blocks_a):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop
            self.blocks = blocks_a

        @torch.no_grad()
        def forward(self, input_ids: torch.LongTensor):
            seq_len = input_ids.size(-1)
            inputs_embeds = self.wte(input_ids)  # word -> vector (ingredients chopped)
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            pos_embeds = self.wpe(position_ids)  # add positional flavor
            hidden_states = inputs_embeds + pos_embeds
            hidden_states = self.drop(hidden_states)
            for block in self.blocks:
                hidden_states = block(hidden_states)[0]  # early cooking steps
            return hidden_states

    # --- Shard B ---
    # Chef who finishes the dish: applies the rest of the layers and returns logits (final flavors)
    class ShardB(nn.Module):
        def __init__(self, blocks_b, ln_f, lm_head):
            super().__init__()
            self.blocks = blocks_b
            self.ln_f = ln_f
            self.lm_head = lm_head

        @torch.no_grad()
        def forward(self, hidden_states: torch.Tensor):
            for block in self.blocks:
                hidden_states = block(hidden_states)[0]  # finish cooking
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)  # final dish prediction
            return logits

    return ShardA(wte, wpe, drop, blocks_a), ShardB(blocks_b, ln_f, lm_head)

# Prepare the two "chefs" (stations)
shard_a, shard_b = split_gpt2_model(model, split_at=SPLIT_AT)
shard_a.to(device).eval()
shard_b.to(device).eval()

# =========================
# API MODELS (Order tickets)
# =========================
# These are like structured order slips given to the kitchen
class InputIDs(BaseModel):
    input_ids: list[int]

class HiddenStates(BaseModel):
    hidden_states: list  # nested lists (batch, seq, hidden_dim)

class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 20

# =========================
# FASTAPI APP (Restaurant front desk)
# =========================
app = FastAPI()

# --- Shard A station ---
@app.post("/forward")
def forward_a(req: InputIDs):
    if SHARD_ROLE != "a":
        return {"error": "This instance is not shard A."}
    # Shard A = prep chef: takes raw IDs and returns partially cooked states
    input_ids = torch.tensor([req.input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = shard_a(input_ids)
    return {"hidden_states": hidden.cpu().tolist()}

# --- Shard B station ---
@app.post("/forward_b")
def forward_b(req: HiddenStates):
    if SHARD_ROLE != "b":
        return {"error": "This instance is not shard B."}
    # Shard B = finishing chef: takes hidden states and outputs logits
    hidden = torch.tensor(req.hidden_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = shard_b(hidden)
    return {"logits": logits.cpu().tolist()}

# --- Coordinator (Head chef) ---
@app.post("/generate")
def generate(req: GenerateReq):
    if SHARD_ROLE != "coordinator":
        return {"error": "This instance is not coordinator."}

    # Step 1: Take customer order (tokenize prompt)
    input_ids = tokenizer.encode(req.prompt)

    # Step 2: For each new token requested, repeat the kitchen workflow
    # - Send current "order" (tokens so far) to Shard A (prep chef).
    # - Shard A partially processes the order (hidden states).
    # - Pass those hidden states to Shard B (finishing chef).
    # - Shard B produces logits (scores for all possible next words).
    # - From these logits, decide the next token to add to the sequence.
    # This loop is like making a dish one ingredient at a time until the recipe is complete.
    for _ in range(req.max_new_tokens):

        # Step 2a: Send order ticket to prep chef (Shard A)
        url_a = f"http://{SHARD_A_SERVICE}:{SHARD_PORT}/forward"
        resp = requests.post(url_a, json={"input_ids": input_ids}, timeout=30)
        resp.raise_for_status()
        hidden = resp.json()["hidden_states"]

        # Step 2b: Send partially cooked dish to finishing chef (Shard B)
        url_b = f"http://{SHARD_B_SERVICE}:{SHARD_PORT}/forward_b"
        resp2 = requests.post(url_b, json={"hidden_states": hidden}, timeout=30)
        resp2.raise_for_status()
        logits = np.array(resp2.json()["logits"])

        # Step 2c: From logits, choose the next token to append
        # Think of this as the head chef tasting the dish and deciding the next ingredient.
        # Instead of always picking the "strongest flavor" (greedy), we sample with some randomness
        # for more natural, varied outputs.
        temperature = 0.6
        top_k = 40

        logits_tensor = torch.tensor(logits[0, -1], dtype=torch.float32)
        logits_tensor = logits_tensor / temperature

        # Apply top-k filtering (restrict to the top 50 likely options)
        # topk_values, topk_indices = torch.topk(logits_tensor, top_k)
        # probs = torch.zeros_like(logits_tensor)
        # probs[topk_indices] = F.softmax(topk_values, dim=-1)
        topk_values, topk_indices = torch.topk(logits_tensor, top_k)
        probs = F.softmax(topk_values, dim=-1)
 



        # Randomly sample the next token from this probability distribution
        #next_token = int(torch.multinomial(probs, num_samples=1))
        next_token = int(topk_indices[torch.multinomial(probs, 1)])
        input_ids.append(next_token)

    # Step 3: Decode full sequence back into text (serve the final dish to the customer)
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return {"generated": text}
