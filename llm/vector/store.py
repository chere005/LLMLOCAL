#!/usr/bin/env python3
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

VECTOR_DIR = None
MODEL_NAME = "BAAI/bge-base-en-v1.5"  # must be cached for offline
embeddings = []
texts = []
tensor_store = None
model = None
store_file = None

# Force CPU
DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # uncomment to enable GPU

def initialize_store(vector_dir):
	global VECTOR_DIR, embeddings, texts, tensor_store, model, store_file
	VECTOR_DIR = vector_dir
	os.makedirs(VECTOR_DIR, exist_ok=True)
	store_file = os.path.join(VECTOR_DIR, "store.pkl")

	# Load existing store
	if os.path.exists(store_file):
		with open(store_file, "rb") as f:
			data = pickle.load(f)
			texts.extend(data.get("texts", []))
			for e in data.get("embeddings", []):
				embeddings.append(torch.tensor(e, device=DEVICE))

	tensor_store = torch.stack(embeddings) if embeddings else torch.empty((0, 768), device=DEVICE)

	# Load sentence-transformers offline
	model = SentenceTransformer(MODEL_NAME, device=DEVICE, local_files_only=True)

def add(text):
	global embeddings, texts, tensor_store
	emb = model.encode(text, convert_to_tensor=True, device=DEVICE)
	embeddings.append(emb)
	texts.append(text)
	tensor_store = torch.stack(embeddings)
	# Save store
	with open(store_file, "wb") as f:
		pickle.dump({"texts": texts, "embeddings": [e.cpu() for e in embeddings]}, f)

def search(query, k=5):
	if not texts:
		return []
	q_emb = model.encode(query, convert_to_tensor=True, device=DEVICE)
	scores = util.cos_sim(q_emb, tensor_store)[0]
	topk = torch.topk(scores, k=min(k, len(texts)))
	return [texts[i] for i in topk.indices.tolist()]