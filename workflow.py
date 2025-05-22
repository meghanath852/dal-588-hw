from langgraph.graph import END, StateGraph

from nodes import (
    retrieve,
    generate,
    grade_documents,
    transform_query,
    decide_to_generate,
    grade_generation_v_documents_and_question,
    web_search_node,
    database_query_node,
    GraphState,
)

def create_workflow(retriever):
    """
    Creates and compiles the workflow.

    Args:
        retriever: The retriever object.

    Returns:
        function: The compiled workflow function.
    """
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("database_query", database_query_node)

    # Build graph
    workflow.set_entry_point("database_query")  # Start with database query check
    
    # Add conditional edge after database_query
    workflow.add_conditional_edges(
        "database_query",
        lambda state: "generate" if state.get("is_db_question", False) and state.get("db_available", False) and any(doc.metadata.get("source") == "postgresql_database" for doc in state["documents"]) else "retrieve",
        {
            "generate": "generate",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    
    # Add conditional edges after generate
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "transform_query",
            "not supported": "transform_query",
        },
    )

    # Compile the workflow
    return workflow.compile()