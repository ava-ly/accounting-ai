"""This file provides tools for an AI agent to interact with a database of accounting laws and perform VAT calculations. """

import os #for accessing environment variables
import logging
from dotenv import load_dotenv #loads env variable from .env file
from supabase import create_client #creates a connection to Supabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings #generates vector embeddings
from langchain_core.tools import tool #decorator to create LangChain tools that AI agents can use
from typing import Optional, Tuple, Union #type hints for better code docs
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Custom Exceptions
class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class DatabaseError(ToolError):
    """Exception for database-related errors."""
    pass

class ValidationError(ToolError):
    """Exception for input validation errors."""
    pass

# Configuration
class ToolConfig:
    """Configuration constants for tools."""
    MATCH_THRESHOLD = 0.5
    MATCH_COUNT = 10
    DEFAULT_VAT_RATE = 8
    MIN_QUERY_LENGTH = 2

# Singleton Database Client
class DatabaseClient:
    """Thread-safe singleton client for database operations."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            DatabaseClient._initialized = True
    
    def _initialize(self):
        """Initialize database connection and embeddings."""
        load_dotenv()

        # Initialize Supabase
        supabase_url = os.getenv("SUPABASE_URL") #get URL from .env
        supabase_key = os.getenv("SUPABASE_KEY") #get key from .env
        if not supabase_url or not supabase_key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        self.supabase = create_client(supabase_url, supabase_key) #create the client

        # Initialize embeddings
        gemini_api_key = os.getenv("GEMINI_API_KEY") #get API key from environment
        if not gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment") # raise error if missing
        #initialize the embeddings model with specific settings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )

    def get_client(self):
        """Get the Supabase client."""
        return self.supabase

    def get_embeddings(self):
        """Get the embeddings model."""
        return self.embeddings

# Helper Functions
def parse_numeric_input(value: Union[str, float, int]) -> float:
    """Parse and clean numeric input from various formats."""
    if isinstance(value, (int, float)):
        return float(value)
    
    # Clean string input
    cleaned = str(value).strip()
    cleaned = cleaned.replace(",", "").replace("_", "")
    cleaned = cleaned.replace("k", "000").replace("K", "000")
    cleaned = cleaned.replace("%", "")

    try:
        return float(cleaned)
    except ValueError:
        raise ValidationError(f"Cannot parse numeric value: {value}")

def format_document_result(doc: dict) -> str:
    """Format a single document result for AI consumption."""
    meta = doc.get('metadata') or {}
    domain = meta.get('domain', 'GENERAL')
    priority = meta.get('priority_label', 'LEGACY')

    return (
        f"[{domain}] [{priority}]\n"
        f"DOCUMENT: {meta.get('official_name', 'Unknown')}\n"
        f"EFFECTIVE: {meta.get('effective_date', 'Unknown')}\n"
        f"CONTENT: {doc.get('content', '')}\n"
        "--------------------------------------------------"
    )

def validate_search_query(func):
    """Decorator to validate search query input."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # The query should be the first argument
        if args:
            query = args[0]
        elif 'query' in kwargs:
            query = kwargs['query']
        else:
            raise ValidationError("Search query not found in arguments")
        
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if len(query.strip()) < ToolConfig.MIN_QUERY_LENGTH:
            raise ValidationError(f"Search query too short (minimum {ToolConfig.MIN_QUERY_LENGTH} characters)")
        
        # Update the query in kwargs
        if 'query' in kwargs:
            kwargs['query'] = query.strip()
            return func(**kwargs)
        else:
            return func(query.strip(), *args[1:])
    return wrapper

# Database Operations
def search_documents(query: str, db_client: DatabaseClient) -> list:
    """Search documents in the database"""
    try:
        embeddings = db_client.get_embeddings()
        query_vector = embeddings.embed_query(query)

        supabase = db_client.get_client()
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_threshold": ToolConfig.MATCH_THRESHOLD,
                "match_count": ToolConfig.MATCH_COUNT
            }
        ).execute()

        # Supabase returns data in response.data
        if hasattr(response, 'data'):
            return response.data
        else:
            # Try to access it directly
            return []
    except Exception as e:
        logger.error(f"Database search error: {e}")
        raise DatabaseError(f"Failed to search documents: {str(e)}")

def format_search_results(documents: list) -> str:
    """Format search results for AI consumption."""
    if not documents:
        return "SYSTEM: No relevant documents found. Do not answer from memory."

    formatted_results = []
    for doc in documents:
        formatted_results.append(format_document_result(doc))

    return "\n".join(formatted_results)

# Tool Definitions
@tool
@validate_search_query
def search_accounting_law(query: str):
    """
    Use this tool to find information about Vietnamese Accounting Laws, 
    Circulars (like Circular 200), Tax regulations, or specific Accounts (TK 111, etc).
    Input should be a search query like "Definition of Account 111".
    """
    logger.info(f"Agent is searching database for: {query}")

    try:
        # Initialize database client
        db_client = DatabaseClient()

        # Search documents
        documents = search_documents(query, db_client)

        # Log results
        if documents:
            logger.info(f"Found {len(documents)} relevant documents")
        else:
            logger.info("No documents found matching the query")

        # Format and return results
        return format_search_results(documents)

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return f"Validation error: {str(e)}"
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return f"Database error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in search_accounting_law: {e}")
        return f"Unexpected error: {str(e)}"

@tool
def calculate_vat(amount: Union[str, float, int], rate_percent: Union[str, float, int] = ToolConfig.DEFAULT_VAT_RATE):
    """
    Use this tool to calculate VAT (Value Added Tax) or generic percentages.
    Input: amount (number), rate_percent (number, default is 8).
    """
    try:
        # Parse inputs
        f_amount = parse_numeric_input(amount)
        f_rate = parse_numeric_input(rate_percent)
        
        # Validate inputs
        if f_amount < 0:
            raise ValidationError("Amount cannot be negative")
        if f_rate < 0 or f_rate > 100:
            raise ValidationError("Rate must be between 0 and 100 percent")
        
        logger.info(f"Agent is calculating VAT: {f_amount} * {f_rate}%")
        
        # Calculate VAT
        tax = f_amount * (f_rate / 100)
        total = f_amount + tax
        
        # Format result
        return f"Amount: {f_amount:,.0f} | Tax ({f_rate}%): {tax:,.0f} | Total: {total:,.0f}"
        
    except ValidationError as e:
        logger.warning(f"Validation error in calculate_vat: {e}")
        return f"Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in calculate_vat: {e}")
        return f"Error: {str(e)}"

# Summary: This file provides two main tools for an AI accounting agent:
# 1. `search_accounting_law`: Performs semantic search in a database of Vietnamese accounting documents using vector embeddings
# 2. `calculate_vat`: Calculates Value Added Tax with flexible input formats
# The tools are designed to be used by an AI agent (through LangChain/LangGraph) to access specific knowledge and perform calculations without relying on the AI's internal knowledge, ensuring accurate, up-to-date information from verified sources.
