import os
import sqlite3
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS


class EnhancedRAGPipeline:
    def __init__(self, db_path="structured_notifications1.db", vector_path="rbi_faiss_structured1.index", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", generation_model_name="google/flan-t5-large"):
        """
        Initialize the RAG pipeline with FAISS, SQLite, and Hugging Face transformers.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        self.db_path = db_path
        self.vector_path = vector_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Initialize Hugging Face tokenizer and model for generation
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)

        # Load or build the FAISS vector store
        self.vector_store = self._load_or_build_vectorstore()

    def _load_or_build_vectorstore(self):
        """
        Load or build the FAISS vector store from the SQLite database.
        """
        try:
            # Define the FAISS index file path
            index_file = os.path.join(self.vector_path, "index.faiss")

            # Check if the FAISS index file exists
            if os.path.exists(index_file):
                self.logger.info("Loading existing FAISS index...")
                return FAISS.load_local(
                    self.vector_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # Enable deserialization if you trust the source
                )
            else:
                self.logger.info("Building FAISS index from database...")
                # Connect to the SQLite database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM notifications")
                docs = [row[0] for row in cursor.fetchall()]
                conn.close()

                if not docs:
                    raise RuntimeError("No documents found in the database to build the FAISS index.")

                # Convert documents into LangChain Document objects
                documents = [Document(page_content=doc) for doc in docs]
                vector_store = FAISS.from_documents(documents, self.embeddings)

                # Ensure the index directory exists
                os.makedirs(self.vector_path, exist_ok=True)
                vector_store.save_local(self.vector_path)
                return vector_store
        except Exception as e:
            self.logger.error(f"Error loading or building FAISS index: {e}")
            raise

    def get_citations(self, context):
        """
        Retrieve unique and relevant citations (URLs or PDF links) for the given context from the SQLite database.
        """
        citations = set()  # Use a set to store unique citations
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Match based on partial content from the context
            cursor.execute("SELECT url, pdf_url FROM notifications WHERE content LIKE ?", (f"%{context[:100]}%",))
            rows = cursor.fetchall()
            for row in rows:
                if row[1]:  # If a PDF URL exists
                    citations.add(row[1])
                if row[0]:  # If a web URL exists
                    citations.add(row[0])
        except Exception as e:
            self.logger.error(f"Error retrieving citations: {e}")
        finally:
            conn.close()
        return list(citations)  # Convert the set back to a list

    def generate_response(self, query, top_k=5, max_length=250):
        """
        Generate a response for the given query using retrieved context and Hugging Face model.
        """
        try:
            # Retrieve the most relevant context
            self.logger.info(f"Retrieving context for query: {query}")
            docs = self.vector_store.similarity_search(query, k=top_k)
            context = " ".join([doc.page_content for doc in docs])

            if not context:
                self.logger.warning("No relevant documents found for the query.")
                return {"Query": query, "Error": "No relevant documents found."}

            # Retrieve citations for the context
            citations = self.get_citations(context)

            # Generate a summary of the retrieved context
            prompt = f"Summarize the following context for the query: '{query}'. Context: {context}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = self.model.generate(inputs.input_ids, max_length=max_length, num_beams=5, early_stopping=True)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Adjust prompt for PDF queries
            if "pdf" in query.lower():
                refined_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"The document '{query.split()[-1]}' contains important details. "
                    f"Provide a comprehensive list of the key points or sections in the PDF. "
                    f"Be exhaustive and structured in the response."
                )
            else:
                refined_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"Generate a detailed and specific response based on the query and context summary. "
                    f"Include actionable insights, examples, and relevant information."
                )

            # Generate detailed response
            inputs = self.tokenizer(refined_prompt, return_tensors="pt", truncation=True, max_length=512)
            response_ids = self.model.generate(inputs.input_ids, max_length=max_length, num_beams=5, early_stopping=True)
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Retry mechanism for incomplete PDF responses
            if "contains important details" in response_text and len(response_text) < 100:
                self.logger.warning("Incomplete response detected for PDF query. Retrying with enhanced prompt.")
                retry_prompt = (
                    f"Query: {query}\n\n"
                    f"Context Summary: {summary}\n\n"
                    f"Provide a detailed summary of the document '{query.split()[-1]}'. Include all key sections, "
                    f"important data points, and any relevant information mentioned in the PDF."
                )
                inputs = self.tokenizer(retry_prompt, return_tensors="pt", truncation=True, max_length=512)
                response_ids = self.model.generate(inputs.input_ids, max_length=max_length, num_beams=5, early_stopping=True)
                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Deduplicate and truncate citations
            unique_citations = citations[:5]  # Limit to 5 citations for readability

            # Return the structured response with citations
            return {
                "Query": query,
                "Context": context[:500],  # Truncate context for readability
                "Summary": summary,
                "Response": response_text,
                "Citations": unique_citations,
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {"Query": query, "Error": f"Failed to generate response due to: {e}"}


if __name__ == "__main__":
    pipeline = EnhancedRAGPipeline(
        db_path="structured_notifications1.db",
        vector_path="rbi_faiss_structured1.index",
    )

    # Expanded list of queries to test the pipeline
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
