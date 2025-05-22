from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_text_into_documents(text, source="input.pdf"):
    """
    Splits text into smaller chunks for processing.

    Args:
        text (str): The text to split.
        source (str): The source of the text (default: "input.pdf").

    Returns:
        List[Document]: A list of Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=50
    )
    # Create documents with source metadata
    doc_splits = text_splitter.create_documents(
        [text],
        metadatas=[{"source": source} for _ in range(len(text_splitter.split_text(text)))]
    )
    return doc_splits