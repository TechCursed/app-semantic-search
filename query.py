import numpy as np
import faiss
import json
import requests

VECTOR_DB = "vector.index"
METADATA_DB = "metadata.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"  # Use any local Ollama model you’ve pulled

def get_ollama_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]

def query_ollama(system_prompt, user_prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            stream=True  # ← allow streaming response
        )
        
        full_reply = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        full_reply += data["message"]["content"]
                except Exception as e:
                    print(f"⚠️ Skipping line due to error: {e}")
        
        return full_reply.strip()

    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        return "Failed to query LLM."


def search_index(query_embedding, k=3):
    index = faiss.read_index(VECTOR_DB)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return I[0]

def load_metadata():
    with open(METADATA_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    question = input("🔎 Ask a question: ")
    embedding = get_ollama_embedding(question)
    top_k = search_index(embedding)
    metadata = load_metadata()
    context = "\n\n".join([metadata[i]["text"] for i in top_k if i < len(metadata)])

    print("🧠 Querying LLM...")
    system_prompt = "You are a helpful assistant. Use the following context to answer user questions."
    final_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    answer = query_ollama(system_prompt, final_prompt)
    print("\n🗣 Answer:")
    print(answer)

if __name__ == "__main__":
    main()
