from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(documents, collection_name="rag-chroma", persist_directory="./chroma_db"):
    """
    Creates and persists a vector store from documents.

    Args:
        documents (List[Document]): The documents to store.
        collection_name (str): Name of the collection.
        persist_directory (str): Directory to persist the vector store.

    Returns:
        Chroma: The vector store object.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore

def get_retriever(vectorstore):
    """
    Gets a retriever from the vector store.

    Args:
        vectorstore (Chroma): The vector store object.

    Returns:
        Retriever: The retriever object.
    """
    retriever = vectorstore.as_retriever()
    return retriever