import requests
import os

def query_with_rag(query):
    # Do simple vector retrieval here from FAISS
    # Then pass to GROQ
    prompt = f"Given this query: '{query}', and the following meeting context, answer accordingly."
    context = "..."  # fetched from FAISS
    full_prompt = prompt + "\n\n" + context
    response = requests.post(
        "https://api.groq.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json={"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": full_prompt}]}
    )
    return response.json()
