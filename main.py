import os
import warnings
import sys

from pdf_utils import extract_text_from_pdf
from document_processing import split_text_into_documents
from vectorstore import create_vectorstore, get_retriever
from workflow import create_workflow

warnings.filterwarnings("ignore")

# Path to the PDF file
pdf_path = "input.pdf"
if not os.path.exists(pdf_path):
    print(f"PDF file '{pdf_path}' not found.")
    sys.exit(1)

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Split the text into Document objects with source metadata
doc_splits = split_text_into_documents(pdf_text, source=pdf_path)

# Add to vector store
vectorstore = create_vectorstore(
    documents=doc_splits,
    collection_name="rag-chroma",
    persist_directory="./chroma_db",
)
retriever = get_retriever(vectorstore)

# Create the workflow
app = create_workflow(retriever)
print("Workflow compiled successfully!")

