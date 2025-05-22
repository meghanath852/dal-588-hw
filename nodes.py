from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from pprint import pprint
from langchain.prompts import ChatPromptTemplate

from prompts import (
    retrieval_grader,
    rag_chain,
    hallucination_grader,
    answer_grader,
    question_rewriter,
)
from websearch import web_search
from database_utils import is_database_question, generate_sql_query, execute_sql_query
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
        is_db_question: whether the question is database-related
        db_available: whether the database is available
    """
    question: str
    generation: str
    documents: List[Document]
    is_db_question: bool
    db_available: bool

def retrieve(state, retriever):
    """
    Retrieve documents.

    Args:
        state (dict): The current graph state.
        retriever: The retriever object.

    Returns:
        dict: Updated state with retrieved documents.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def web_search_node(state):
    """
    Perform web search when no relevant documents are found.
    
    Args:
        state (dict): The current graph state.
        
    Returns:
        dict: Updated state with web search results as documents.
    """
    print("---WEB SEARCH---")
    # Invoke the web_search function from websearch.py
    web_search_results = web_search(state)
    
    # Return the updated state with web search results
    return web_search_results

def generate(state):
    """
    Generate an answer.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with the generated answer.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Format documents
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)

    # RAG generation
    generation = rag_chain.invoke({"context": formatted_docs, "question": question})
    
    # Log the generated answer
    print("\n---GENERATED ANSWER---")
    print(generation)
    print("----------------------\n")
    
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Grade the relevance of documents to the question.
    """
    print("---GRADE DOCUMENTS---")
    
    question = state["question"]
    documents = state["documents"]
    
    # Grade each document
    filtered_docs = []
    for doc in documents:
        # Always mark DB results as relevant
        if doc.metadata.get("source") == "postgresql_database":
            print("---GRADE: DATABASE RESULT FOUND, MARKING AS RELEVANT---")
            filtered_docs.append(doc)
            continue
        
        # For non-database documents, use the retrieval grader
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        print(f"---GRADE: DOCUMENT {'RELEVANT' if grade.lower() == 'yes' else 'NOT RELEVANT'}---")
        
        if grade.lower() == "yes":
            filtered_docs.append(doc)
    
    return {"documents": filtered_docs}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with the rephrased question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def database_query_node(state):
    """
    Check if question is database-related and execute SQL query if it is.
    
    Args:
        state (dict): The current graph state.
        
    Returns:
        dict: Updated state with database results if applicable.
    """
    print("---CHECKING DATABASE RELEVANCE---")
    question = state["question"]
    
    # Check if database is available
    db_available = state.get("db_available", False)
    if not db_available:
        print("---DATABASE NOT AVAILABLE, SKIPPING DATABASE QUERY---")
        return {
            "documents": state.get("documents", []),
            "question": question,
            "is_db_question": False,
            "db_available": False
        }
    
    # Check if question is database-related
    try:
        if is_database_question(question):
            print("---QUESTION IS DATABASE-RELATED---")
            
            # Generate SQL query
            sql_query = generate_sql_query(question)

            print(f"---GENERATED SQL QUERY---\n{sql_query}")
            
            if sql_query:
                print(f"---EXECUTING SQL QUERY---\n{sql_query}")
                
                try:
                    # Execute query and get results as Document
                    db_doc = execute_sql_query(sql_query, question)
                    
                    # Check if the result is valid (not an error)
                    if db_doc.page_content.startswith("Error"):
                        print("---SQL EXECUTION ERROR, IGNORING DB RESULT---")
                        return {
                            "documents": state.get("documents", []),
                            "question": question,
                            "is_db_question": False,
                            "db_available": True
                        }
                    
                    # Add database document to existing documents
                    documents = state.get("documents", [])
                    documents.append(db_doc)
                    
                    print("---DATABASE RESULT ADDED TO DOCUMENTS---")
                    return {
                        "documents": documents,
                        "question": question,
                        "is_db_question": True,
                        "db_available": True
                    }
                except Exception as e:
                    logger.error(f"Error executing SQL query: {str(e)}")
                    print(f"---ERROR EXECUTING SQL QUERY: {str(e)}---")
                    return {
                        "documents": state.get("documents", []),
                        "question": question,
                        "is_db_question": False,
                        "db_available": False  # Mark database as unavailable after error
                    }
            else:
                print("---NO SQL QUERY GENERATED---")
                return {
                    "documents": state.get("documents", []),
                    "question": question,
                    "is_db_question": False,
                    "db_available": True
                }
        else:
            print("---QUESTION IS NOT DATABASE-RELATED---")
            return {
                "documents": state.get("documents", []),
                "question": question,
                "is_db_question": False,
                "db_available": True
            }
    except Exception as e:
        logger.error(f"Error in database query node: {str(e)}")
        print(f"---ERROR IN DATABASE QUERY NODE: {str(e)}---")
        return {
            "documents": state.get("documents", []),
            "question": question,
            "is_db_question": False,
            "db_available": False  # Mark database as unavailable after error
        }

def decide_to_generate(state):
    """
    Decide whether to generate an answer or rephrase the question.

    Args:
        state (dict): The current graph state.

    Returns:
        str: Decision for the next node to call.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    is_db_question = state.get("is_db_question", False)

    # If it's a database question and we have results, go straight to generate
    if is_db_question and any(doc.metadata.get("source") == "postgresql_database" for doc in filtered_documents):
        print("---DECISION: DATABASE RESULTS FOUND, GENERATE---")
        return "generate"

    if not filtered_documents:
        # Check if we've already tried web search
        tried_web_search = state.get("tried_web_search", False)
        
        if tried_web_search:
            print("---DECISION: NO RELEVANT DOCUMENTS AND WEB SEARCH ALREADY TRIED, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: NO RELEVANT DOCUMENTS, TRY WEB SEARCH---")
            return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determine whether the generation is grounded in the documents and answers the question.

    Args:
        state (dict): The current graph state.

    Returns:
        str: Decision for the next node to call.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Format documents
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)

    score = hallucination_grader.invoke({"documents": formatted_docs, "generation": generation})
    grade = score.binary_score

    if grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check if generation addresses the question
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"