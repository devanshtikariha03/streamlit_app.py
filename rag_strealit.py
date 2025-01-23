import streamlit as st
from anotherbest import EnhancedRAGPipeline

# Initialize the RAG pipeline
pipeline = EnhancedRAGPipeline(
    db_path="structured_notifications.db",
    vector_path="rbi_faiss_structured.index",
)

def generate_response(query):
    """Process user query using the RAG pipeline and return summary, response, and citations."""
    try:
        # Set consistent max_length for summaries
        max_length = 250
        response = pipeline.generate_response(query, max_length=max_length)
        summary = response.get("Summary", "No summary available.")
        response_text = response.get("Response", "No response available.")
        citations = response.get("Citations", [])
        citations_text = "\n".join(citations) if citations else "No citations available."

        # Combine summary, response, and citations into one output
        combined_output = (
            f"### Summary\n{summary}\n\n"
            f"### Response\n{response_text}\n\n"
            f"### Citations\n{citations_text}"
        )
        return combined_output
    except Exception as e:
        return f"Error: {e}"

# Streamlit Interface
def main():
    st.title("RBI Notifications Query System")
    st.markdown("Easily query RBI notifications and regulations.")

    # Predefined queries
    queries = [
        "What is the role of NaBFID in financial markets as regulated by the RBI?",
        "Can you summarize the recent RBI notification on the creation of new districts in Nagaland?",
        "What are the RBI guidelines for assigning lead bank responsibilities in newly created districts?",
        "How does the RBI regulate All-India Financial Institutions (AIFIs)?",
        "What are the Basel III capital framework guidelines issued by the RBI?",
        "What are the RBI’s instructions regarding credit default swaps?",
        "How does the RBI regulate repo transactions in the financial markets?",
        "Which bank has been assigned lead responsibility for the new district of Meluri in Nagaland?",
        "Are there changes in lead bank responsibilities for other districts in Nagaland?",
        "What legal provisions empower the RBI to regulate NaBFID as an AIFI?",
        "Under what sections of the Reserve Bank of India Act, 1934, does the RBI issue guidelines for financial institutions?",
        "What are the RBI directions for repurchase transactions as per the updated guidelines of 2018?",
        "How does the RBI ensure compliance with the Prudential Regulations on Basel III for financial institutions?",
        "What changes were introduced in the RBI’s notification dated January 01, 2025?",
        "Explain the significance of the Master Direction – Reserve Bank of India (Credit Derivatives) Directions, 2022.",
    ]

    # User Input
    st.sidebar.header("Query Options")
    predefined_query = st.sidebar.selectbox("Select a predefined query", options=[""] + queries)
    custom_query = st.sidebar.text_input("Or type your query", "")

    # Query submission
    if st.sidebar.button("Submit"):
        query = predefined_query if predefined_query else custom_query
        if query:
            with st.spinner("Processing your query..."):
                output = generate_response(query)
            st.markdown(output)
        else:
            st.error("Please select or enter a query.")

if __name__ == "__main__":
    main()
