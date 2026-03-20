#!/usr/bin/env python3
import sys
import os
import subprocess

# Ensure vector store is in path
VECTOR_DIR = os.path.expanduser("")
sys.path.append(VECTOR_DIR)

from store import add, search, initialize_store

# llama.cpp binary
LLAMA_BIN = os.path.expanduser("")

# GGUF model file
MODEL_PATH = os.path.expanduser("")

# Initialize vector store
initialize_store(VECTOR_DIR)

def run_llama(prompt):
    try:
        result = subprocess.run([
            LLAMA_BIN,
            "--model", MODEL_PATH,
            "--threads", "8",
            "--prompt", prompt,
            "--n-predict", str(2**15),
            "--ctx-size", str(2**12)
        ], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running llama: {e.stderr.strip()}"

def get_memory(query, k=5):
    return "\n".join(search(query, k))

def main():
    if len(sys.argv) < 2:
        print("Usage: python llm.py 'your message here'")
        return

    msg = " ".join(sys.argv[1:])

    # Store memory if user wants
    if msg.lower().startswith("remember this:"):
        fact = msg[len("remember this:"):].strip()
        add(fact)
        print("memory stored.")
        return

    # Retrieve memory
    mem = get_memory(msg)
    if mem:
        print("memory:\n", mem)

    # Run llama.cpp
    output = run_llama(msg)
    print(output)

if __name__ == "__main__":
    main()