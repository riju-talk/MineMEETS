# rag_agent.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from groq import Groq

class MeetingAnalyzer:
    def __init__(self, groq_api_key, faiss_index):
        self.client = Groq(api_key=groq_api_key)
        self.retriever = faiss_index.as_retriever()  # FAISS -> LangChain retriever
        
    def _run_llm(self, prompt):
        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
        
    def query(self, question):
        relevant_docs = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        Context: {context}
        
        Question: {question}
        
        Instructions:
        1. Use only information from the context
        2. Format answers clearly
        3. Highlight key action items
        """
        
        return self._run_llm(prompt)