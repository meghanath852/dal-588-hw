from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Define grading schemas
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Retrieval grader prompt
retrieval_grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader_llm = llm.with_structured_output(GradeDocuments)
retrieval_grader = retrieval_grader_prompt | retrieval_grader_llm

# RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

# Hallucination grader prompt
hallucination_grader_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader_llm = llm.with_structured_output(GradeHallucinations)
hallucination_grader = hallucination_grader_prompt | hallucination_grader_llm

# Answer grader prompt
answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader_llm = llm.with_structured_output(GradeAnswer)
answer_grader = answer_grader_prompt | answer_grader_llm

# Question rewriter prompt
question_rewriter_system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
question_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_rewriter_system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()