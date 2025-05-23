from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingest import extract_texts_from_txt, embed_and_store
from query import get_ollama_embedding, search_index, load_metadata, query_ollama

app = FastAPI()

# Request models
class QueryRequest(BaseModel):
    question: str

@app.post("/ingest/")
def ingest_documents():
    docs = extract_texts_from_txt("docs")
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found in 'docs/'")
    embed_and_store(docs)
    return {"status": "success", "message": f"{len(docs)} documents ingested and indexed"}

@app.post("/query/")
def query_documents(request: QueryRequest):
    try:
        embedding = get_ollama_embedding(request.question)
        top_k_indices = search_index(embedding)
        metadata = load_metadata()

        # Build context from top results
        context = "\n\n".join([metadata[i]["text"] for i in top_k_indices if i < len(metadata)])
        system_prompt = "You are a helpful assistant. Use the following context to answer user questions."
        final_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"

        answer = query_ollama(system_prompt, final_prompt)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))