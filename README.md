# LLM Sharding Concept

**A hands-on demo to learn the concept of sharding large language models (LLMs)**

This repository demonstrates how to split a Tiny GPT-2 model into multiple “shards” and orchestrate token generation using a **coordinator**. The goal is educational: to help you understand model sharding and distributed inference in a simple setup.

---

## 🏗️ Overview

We split the Tiny GPT-2 model into two shards:

- **Shard A** – the “prep chef”: handles embeddings and early transformer blocks.
- **Shard B** – the “finishing chef”: handles the remaining transformer blocks and produces final logits.
- **Coordinator** – the “head chef”: orchestrates generation by sending prompts to Shard A and hidden states to Shard B.

The architecture mimics a kitchen workflow:

```
User prompt
    │
    ▼
Coordinator (tokenizes prompt)
    │
    ▼
Shard A (partial forward pass)
    │
    ▼
Shard B (finishes forward pass → logits)
    │
    ▼
Coordinator (selects next token & decodes text)
    │
    ▼
Generated text
```

---

## ⚙️ Requirements

- Python 3.10+
- PyTorch
- Transformers
- FastAPI
- Docker & Minikube (for local Kubernetes setup)
- Requests (for notebook client)

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/llm-sharding-demo.git
cd llm-sharding-demo
```

### 2. Build and run Docker images

This example uses **Minikube** for a local Kubernetes cluster:

```bash
# Install Minikube
brew install minikube

# Install Docker Desktop
brew install --cask docker

# Start Kubernetes cluster
minikube start --driver=docker --cpus=8 --memory=16g

# Enable metrics server (optional)
minikube addons enable metrics-server
```

Build the Docker image and deploy:

```bash
# Ensure commands run inside Minikube Docker env
eval $(minikube docker-env)

# Build image
docker build -t llm-sharding:latest .

# Deploy shards and coordinator to Kubernetes
kubectl apply -f k8s/
kubectl get pods -o wide
kubectl get svc
```

---

### 3. Run the coordinator

Before sending prompts, port-forward the coordinator service:

```bash
kubectl port-forward svc/llm-coordinator 5000:5000
```

---

### 4. Test generation using the notebook

The included Jupyter notebook demonstrates how to:

- Send a prompt to the coordinator
- Receive generated output
- Iterate token-by-token via Shard A and Shard B

Example usage:

```python
prompt = "Hi, "
output = generate_text(prompt, max_new_tokens=20)
print("Prompt:", prompt)
print("Generated output:", output)
```

---

## 📚 Learning Goals

- Understand **model sharding** for LLMs.
- Learn how to orchestrate multiple shards via a **coordinator**.
- Explore **distributed inference workflows** locally with Kubernetes.

---

## 📝 Notes

- This demo uses **CPU** for simplicity.
- Tiny GPT-2 is used to keep the example lightweight; results may not be coherent due to small model size.
- For more meaningful text, try larger models or avoid splitting very small models.
- The notebook demonstrates **single-shell execution** to avoid Docker/Minikube environment issues.

---

## 📁 Repository Structure

```
.
├─ server.py         # FastAPI server for Shard A, Shard B, and Coordinator
├─ requirements.txt
├─ Dockerfile        # Docker build file for the service
├─ notebook.ipynb    # Jupyter notebook to test generation
├─ README.md         # This README
└─ k8s/              # Kubernetes deployment YAMLs for coordinator and shards
   ├─ shard-a-deployment.yaml
   ├─ shard-b-deployment.yaml
   ├─ shard-a-service.yaml
   ├─ shard-b-service.yaml
   ├─ coordinator-deployment.yaml
   └─ coordinator-service.yaml
```
