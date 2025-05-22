import streamlit as st
from main import app  # Import the workflow
from database_utils import load_ipl_data  # Import database loading function
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page setup
st.set_page_config(page_title="RAG Q&A", layout="centered")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    - Enter a question related to your document or IPL data.
    - The app will use RAG (Retrieval-Augmented Generation) with Self Reflection to retrieve and generate an answer.
    - If no relevant documents are found in the database, web search will be used.
    - For IPL-related questions, the app will query the PostgreSQL database directly.
    """
)
st.sidebar.markdown(
    """
    ---
    ### **V. Meghanath Reddy**
    *21119054*
    ### **Arpan Kumar**
    *21322009*
    ### **S. S. H . Quadri**
    *21119049*
    
    ---
    """
)

# Initialize database if not already done
@st.cache_resource
def init_database():
    try:
        load_ipl_data()
        st.session_state.db_available = True
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        st.session_state.db_available = False
        return False

# Main interface
st.title("RAG Workflow Q&A")

# Initialize session state for database availability if not already set
if 'db_available' not in st.session_state:
    st.session_state.db_available = False

# Initialize database
db_status = init_database()
if db_status:
    st.success("IPL database initialized successfully!")
else:
    st.warning("IPL database not available. The app will continue with document search and web search only.")

# User input
user_question = st.text_input("Your Question:", value="How many runs did V Kohli score")

# Submit button
if st.button("Get Answer") and user_question.strip():
    # Run the RAG workflow
    inputs = {"question": user_question, "db_available": st.session_state.db_available}
    final_generation = None
    
    try:
        spinner_placeholder = st.empty()
        spinner_placeholder.text("Processing...")

        # Display a loading spinner
        iteration = 1
        with st.spinner(""):
            for output in app.stream(inputs, {"recursion_limit": 8}):
                for key, value in output.items():
                    xx = len(value["documents"])
                    if(key == "transform_query"): iteration += 1
                    if(key == "web_search"): spinner_placeholder.text(f"Iteration = {iteration} (Web Search)")
                    elif(key == "database_query"): spinner_placeholder.text(f"Iteration = {iteration} (Database Query)")
                    else: spinner_placeholder.text(f"Iteration = {iteration}")
                    

        spinner_placeholder.text(f"")
        # Display the final answer
        if value["generation"]:
            st.success("Answer:")
            st.write(value["generation"])
            
            # Display sources
            st.subheader("Sources:")
            for i, doc in enumerate(value["documents"]):
                source = doc.metadata.get("source", "Unknown")
                st.write(f"**Source {i+1}:** {source}")
                
                # Display database query if present
                if source == "postgresql_database":
                    st.info("This answer includes data from the IPL database.")
                    with st.expander("View SQL Query"):
                        st.code(doc.metadata.get("query", "No query available"), language="sql")
                
                # Display web search info if present
                elif source == "tavily_web_search":
                    st.info("This answer was supplemented with web search results.")
                
                # Display document content if from vector database (PDF)
                elif source.endswith(".pdf"):
                    with st.expander("View Document Content"):
                        st.text(doc.page_content)
        else:
            st.error("Sorry, I didn't understand your question. Do you want to connect with a live agent?")
    
    except Exception as e:
        spinner_placeholder.text(f"")
        st.error(f"Sorry, I didn't understand your question. Do you want to connect with a live agent? Error: {str(e)}")