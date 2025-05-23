import os
import numpy as np
import faiss
import json
from tqdm import tqdm
import hashlib

EMBED_MODEL = "nomic-embed-text"  # Ollama-supported embedding model
VECTOR_DB = "vector.index"
METADATA_DB = "metadata.json"
DOCS_DIR = "docs"

# ---- 1. Read .txt Files ----
def extract_texts_from_txt(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            try:
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        docs.append({"filename": fname, "text": text})
                    else:
                        print(f"⚠️ Empty file: {fname}")
            except Exception as e:
                print(f"❌ Failed to read {fname}: {e}")
    return docs

# ---- 2. Split into Chunks ----
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
    return chunks

# ---- 3. Get Embedding from Ollama ----
def get_ollama_embedding(text):
    import requests
    import json

    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": EMBED_MODEL,  # should be "nomic-embed-text"
                "prompt": text
            })
        )

        data = response.json()

        if "embedding" in data:
            return data["embedding"]
        else:
            print("❌ Unexpected response:", data)
            return None

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


# ---- 4. Store Embeddings in FAISS ----
def embed_and_store(docs):
    all_chunks = []
    metadata = []
    embeddings = []

    for doc in tqdm(docs, desc="🔍 Embedding chunks"):
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            embedding = get_ollama_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
                metadata.append({
                    "text": chunk,
                    "source": doc["filename"]
                })

    if not embeddings:
        print("❌ No embeddings were created. Exiting.")
        return

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, VECTOR_DB)

    with open(METADATA_DB, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"✅ Stored {len(embeddings)} embeddings.")

# ---- MAIN ----
def main():
    print("📄 Reading .txt files...")
    docs = extract_texts_from_txt(DOCS_DIR)
    if not docs:
        print("❌ No documents found.")
        return
    embed_and_store(docs)

if __name__ == "__main__":
    main()
