import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Database connection parameters
DB_NAME = os.getenv("DB_NAME", "ipl_data")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Initialize OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Database schema information
DB_SCHEMA = """
Table: ipl_deliveries
Columns:
- match_id: Match identifier
- inning: Inning number (1 or 2)
- batting_team: Team that is batting
- bowling_team: Team that is bowling
- over: Over number
- ball: Ball number within the over
- batter: Batsman's name
- bowler: Bowler's name
- non_striker: Non-striker's name
- batsman_runs: Runs scored by the batsman
- extra_runs: Extra runs (wides, no-balls, etc.)
- total_runs: Total runs in the delivery
- extras_type: Type of extra (wide, no-ball, etc.)
- is_wicket: Whether a wicket was taken (1 or 0)
- player_dismissed: Name of dismissed player
- dismissal_kind: Type of dismissal
- fielder: Fielder involved in dismissal
"""

def create_database():
    """Create the database if it doesn't exist"""
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
    
    cursor.close()
    conn.close()

def load_ipl_data(csv_path="deliveries.csv"):
    """Load IPL data from CSV into PostgreSQL"""
    # Create database if it doesn't exist
    create_database()
    
    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Load data into PostgreSQL
        df.to_sql('ipl_deliveries', engine, if_exists='replace', index=False)
        
        # Create indexes for common query patterns
        with engine.connect() as conn:
            # Execute each CREATE INDEX statement using text()
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_id ON ipl_deliveries(match_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_batter ON ipl_deliveries(batter)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_bowler ON ipl_deliveries(bowler)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_batting_team ON ipl_deliveries(batting_team)"))
            conn.commit()
            
        print("Successfully loaded IPL data and created indexes")
        return True
    except Exception as e:
        print(f"Error loading IPL data: {str(e)}")
        return False

def is_database_question(question):
    """Use OpenAI to determine if the question is related to IPL database"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a database expert. Given a question, determine if it can be answered using the IPL cricket database.
        Here is the database schema:
        {DB_SCHEMA}
        
        Respond with ONLY 'yes' or 'no'."""),
        ("human", question)
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return response.content.strip().lower() == 'yes'

def generate_sql_query(question):
    """Use OpenAI to generate SQL query from natural language question"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a SQL expert. Generate a PostgreSQL query to answer the question using the IPL database.
        Here is the database schema:
        {DB_SCHEMA}
        
        Rules:
        1. Only use columns that exist in the schema
        2. Return the query only, no explanations or triple backticks or language name
        3. Format the query for readability
        4. The player name should be in the format of 'P Name'. For example, 'V Kohli' instead of 'Virat Kohli' and 'MS Dhoni' instead of 'Mahendra Singh Dhoni'.
        5. If the question is not relevant to the database, return 'None'"""),
        ("human", question)
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    query = response.content.strip()
    
    # Validate that the response is a SQL query
    print(f"Generated query: {query}")
    if query.lower() == 'none' or not query.lower().startswith('select'):
        return None
        
    return query

def execute_sql_query(query, question):
    """Execute SQL query and return results as a Document"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        # Print results
        print("\n---SQL QUERY RESULTS---")
        print(df.to_string())
        print("------------------------\n")
        
        # Close connection
        conn.close()
        
        # Convert results to Q&A format
        result_str = f"The answer for the question '{question}' is: {df.to_string()}"
        
        # Create Document with Q&A format
        doc = Document(
            page_content=result_str,
            metadata={
                "source": "postgresql_database",
                "query": query,
                "rows_returned": len(df)
            }
        )
        
        return doc
    except Exception as e:
        # Print error details
        print(f"\n---SQL EXECUTION ERROR DETAILS---")
        print(f"Error: {str(e)}")
        print("--------------------------------\n")
        
        # Return error as Document
        return Document(
            page_content=f"Error executing query: {str(e)}",
            metadata={
                "source": "postgresql_database_error",
                "query": query
            }
        ) 