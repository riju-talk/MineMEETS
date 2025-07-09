from backend.rag_agent import query_with_rag

def extract_tasks():
    return query_with_rag("Extract all tasks, action items, deadlines and responsible persons from this meeting.")
