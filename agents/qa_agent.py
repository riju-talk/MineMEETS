"""QA Agent for answering questions about meetings using vector database (HF/Unsloth, no OpenAI)."""
from typing import Dict, Any, List, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import your PineconeDB wrapper (try both typical locations)
try:
    from vector_store.pinecone_store import PineconeDB  # preferred location
except Exception:
    from ..pinecone_db import PineconeDB  # if your project keeps it under agents/

from .base_agent import BaseAgent, AgentResponse


class QAAgent(BaseAgent):
    """
    Agent for handling question-answering about meetings.
    - Uses HF Transformers (Unsloth/GPT-OSS) with 4-bit quantization (bitsandbytes).
    - Uses Pinecone retriever (LangChain VectorStore).
    """

    def __init__(self, pinecone_db: PineconeDB):
        super().__init__(
            name="qa_agent",
            description="Answers questions about meeting content using vector database"
        )
        self.pinecone_db = pinecone_db

        # ---- Load quantized OSS model (Unsloth or any HF instruct model) ----
        # Choose model via env; defaults to Unsloth 8B Instruct 4-bit
        model_id = os.getenv("LLM_MODEL_ID", "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit")
        max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

        # 4-bit config (works on GPU; will fallback to CPU if no CUDA)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_cfg,   # quantized weights
            torch_dtype=torch.bfloat16     # compute dtype
        )

        gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            return_full_text=False
        )

        # Wrap HF pipeline for LangChain
        self.llm = HuggingFacePipeline(pipeline=gen)

        # ---- Build retriever & QA chain ----
        self.retriever = self._create_retriever(k=int(os.getenv("RAG_TOP_K", "5")))
        self.qa_chain = self._create_qa_chain()

        # Optional: sanity-check embedding dims vs index dims
        self._sanity_check_dims()

    # ---------- internals ----------
    def _create_retriever(self, k: int = 5):
        """Create a Pinecone retriever with default similarity search."""
        # Expose a get_retriever on your DB so signature matches
        return self.pinecone_db.get_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def _create_qa_chain(self):
        """Create a QA chain with a focused prompt."""
        prompt_template = (
            "You are a helpful meeting assistant. Use ONLY the context to answer.\n"
            "If answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer with citations like [1], [2] where relevant."
        )
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def _sanity_check_dims(self):
        """Warn if Pinecone index dim != embedding model dim."""
        try:
            idx_desc = self.pinecone_db.pc.describe_index(self.pinecone_db.index_name)
            pinecone_dim = getattr(idx_desc, "dimension", None) or getattr(idx_desc, "spec", {}).get("dimension")
            emb_dim = getattr(self.pinecone_db, "dim", None)
            if pinecone_dim and emb_dim and pinecone_dim != emb_dim:
                # Don't crash, but warn loudly in logs
                print(f"[WARN] Pinecone index dim={pinecone_dim} != embedding dim={emb_dim}. Fix this mismatch!")
        except Exception as _:
            pass

    # ---------- public API ----------
    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Args:
            question: user question
            context: optional dict such as {"meeting_id": "M-123"}
        """
        try:
            # Apply per-query metadata filters (e.g., restrict to a meeting)
            if context and "meeting_id" in context:
                self.retriever.search_kwargs["filter"] = {"meeting_id": context["meeting_id"]}

            result = self.qa_chain({"query": question})

            return AgentResponse(
                success=True,
                content={
                    "answer": result["result"],
                    "sources": [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in result["source_documents"]
                    ]
                }
            )
        except Exception as e:
            return AgentResponse(success=False, content=f"Error processing question: {e}")

    def add_meeting_context(self, meeting_id: str, documents: List[Dict[str, Any]]):
        """
        Upsert meeting docs into Pinecone via your DB.
        documents: List[{"text": "...", "metadata": {...}}]
        """
        texts = [d["text"] for d in documents]
        metadatas = [{"meeting_id": meeting_id, **(d.get("metadata") or {})} for d in documents]
        ids = [f"{meeting_id}-{i}" for i in range(len(texts))]
        # Idempotent upsert; existing ids are skipped
        self.pinecone_db.upsert_texts(texts=texts, metadatas=metadatas, ids=ids)
