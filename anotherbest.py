import os
import sqlite3
import logging
from typing import List

# Transformers / LangChain
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EnhancedRAGPipeline:
    """
    A production-oriented Retrieval-Augmented Generation pipeline that:
      1) Loads / builds a FAISS index from an SQLite database (notifications).
      2) Uses a stronger embedding model (default: 'multi-qa-mpnet-base-dot-v1').
      3) Splits documents into smaller chunks for more granular retrieval.
      4) Generates answers using a chosen HF Seq2Seq model (default: 'google/flan-t5-large').
      5) Handles PDF-related queries with specialized prompts.
    """

    def __init__(
        self,
        db_path: str = "structured_notifications1.db",
        vector_path: str = "rbi_faiss_structured2.index",
        embedding_model_name: str = "multi-qa-mpnet-base-dot-v1",
        generation_model_name: str = "google/flan-t5-large",
        chunk_size: int = 500,  # ~500 tokens per chunk
        chunk_overlap: int = 50,
        device: str = None,
    ):
        """
        :param db_path: Path to your SQLite database
        :param vector_path: Directory to store the FAISS index
        :param embedding_model_name: Hugging Face embedding model
        :param generation_model_name: Hugging Face Seq2Seq model for generation
        :param chunk_size: Number of tokens (approx) per document chunk
        :param chunk_overlap: Overlap in tokens between chunks
        :param device: 'cuda' or 'cpu' (auto-detect GPU if None)
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        self.db_path = db_path
        self.vector_path = vector_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Decide on device (CPU or GPU)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Hugging Face embeddings (for retrieval)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Seq2Seq model (for answer generation)
        self.logger.info(f"Loading generation model '{generation_model_name}' on device {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)
        self.model.to(device)

        # Load or build the FAISS vector store
        self.vector_store = self._load_or_build_vectorstore()

    def _load_all_docs_from_db(self) -> List[Document]:
        """
        Fetch all 'content' from 'notifications' table. 
        Convert each row to a Document, then split into smaller chunks.
        """
        self.logger.info(f"Loading data from DB at {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM notifications")
        rows = cursor.fetchall()
        conn.close()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        docs = []
        total_chunks = 0
        for row in rows:
            content = row[0] if row[0] else ""
            # Split into chunks for better retrieval
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                # Each chunk is a separate Document
                docs.append(Document(page_content=chunk))
            total_chunks += len(chunks)

        self.logger.info(f"Loaded {len(rows)} rows from DB, created {total_chunks} chunks.")
        return docs

    def _load_or_build_vectorstore(self) -> FAISS:
        """
        Load or build the FAISS vector store from the DB's chunked documents.
        """
        try:
            index_file = os.path.join(self.vector_path, "index.faiss")

            if os.path.exists(index_file):
                self.logger.info("Loading existing FAISS index...")
                return FAISS.load_local(
                    self.vector_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                self.logger.info("Building FAISS index from database content...")
                docs = self._load_all_docs_from_db()
                if not docs:
                    raise RuntimeError("No documents found in DB to build the FAISS index.")

                vector_store = FAISS.from_documents(docs, self.embeddings)
                os.makedirs(self.vector_path, exist_ok=True)
                vector_store.save_local(self.vector_path)
                return vector_store
        except Exception as e:
            self.logger.error(f"Error loading or building FAISS index: {e}")
            raise

    def get_citations(self, context: str):
        """
        Retrieve relevant citations (URLs, PDF links) from the DB by matching partial content.
        This is a naive approach that looks for a substring match on the first 100 chars.
        """
        citations = set()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT url, pdf_url FROM notifications WHERE content LIKE ?",
                (f"%{context[:100]}%",),
            )
            rows = cursor.fetchall()
            for (url, pdf_url) in rows:
                if pdf_url:
                    citations.add(pdf_url)
                if url:
                    citations.add(url)
        except Exception as e:
            self.logger.error(f"Error retrieving citations: {e}")
        finally:
            conn.close()
        return list(citations)

    def generate_response(self, query: str, top_k=5, max_length=250):
        """
        Retrieve top_k relevant chunks from FAISS, generate summary and final answer with the HF model.
        If the query mentions "pdf", use a specialized prompt to list key points/sections.
        """
        try:
            self.logger.info(f"Retrieving context for query: {query}")
            # Retrieve top-k relevant chunks
            docs = self.vector_store.similarity_search(query, k=top_k)
            context = " ".join([doc.page_content for doc in docs])

            if not context.strip():
                self.logger.warning("No relevant documents found for the query.")
                return {"Query": query, "Error": "No relevant documents found."}

            # Get citations for the context
            citations = self.get_citations(context)

            # Summarize context
            prompt = f"Summarize the following context for the query: '{query}'. Context: {context}"
            summary = self._generate_text(prompt, max_length)

            # Adjust prompt if PDF query
            if "pdf" in query.lower():
                refined_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"The document '{query.split()[-1]}' might have key details. "
                    f"Provide a structured list of the main sections or points in the PDF. "
                    f"Be exhaustive, clear, and concise."
                )
            else:
                refined_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"Generate a detailed and specific response based on the query and context summary. "
                    f"Include actionable insights, examples, and relevant information."
                )

            # Generate final response
            response_text = self._generate_text(refined_prompt, max_length)

            # Retry mechanism for incomplete PDF responses
            if "contains important details" in response_text and len(response_text) < 100:
                self.logger.warning("Incomplete PDF response. Retrying with an enhanced prompt.")
                retry_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"Provide a more comprehensive summary of the document '{query.split()[-1]}'. "
                    f"Include all key sections, data points, and relevant references."
                )
                response_text = self._generate_text(retry_prompt, max_length)

            unique_citations = citations[:5]  # limit to 5 for readability

            return {
                "Query": query,
                "Context": context[:500],  # truncated for readability
                "Summary": summary,
                "Response": response_text,
                "Citations": unique_citations,
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {"Query": query, "Error": f"Failed to generate response due to: {e}"}

    def _generate_text(self, prompt: str, max_length: int):
        """
        Helper to run the HF model and decode the output text. 
        Applies some safe defaults (num_beams=5, early_stopping=True).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage in production-like environment
    pipeline = EnhancedRAGPipeline(
        db_path="structured_notifications1.db",
        vector_path="rbi_faiss_structured2.index",
        embedding_model_name="multi-qa-mpnet-base-dot-v1",  # stronger embeddings than MiniLM
        generation_model_name="google/flan-t5-large",        # can be upgraded to flan-t5-xxl if you have the GPU memory
        chunk_size=500,                                      # chunk docs ~500 tokens
        chunk_overlap=50,
        device=None,  # auto-detect GPU if available
    )

    queries = [
        "What is the role of NaBFID in financial markets as regulated by the RBI?",
        "Can you summarize the recent RBI notification on the creation of new districts in Nagaland?",
        "What are the key details mentioned in the Utkarsh30122022.pdf?",
        "How does the RBI regulate All-India Financial Institutions (AIFIs)?",
        "What are the Basel III capital framework guidelines issued by the RBI?",
        "What are the RBI’s instructions regarding credit default swaps?",
        "Which bank has been assigned lead responsibility for the new district of Meluri in Nagaland?",
        "What legal provisions empower the RBI to regulate NaBFID as an AIFI?",
    ]

    for query in queries:
        print(f"\nProcessing Query: {query}")
        response = pipeline.generate_response(query)
        for key, value in response.items():
            print(f"{key}: {value}")
