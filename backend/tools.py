"""This file provides tools for an AI agent to interact with a database of accounting laws and perform VAT calculations. """

import os #for accessing environment variables
import logging #logging messages at different severity levels
from dotenv import load_dotenv #loads env variable from .env file
from supabase import create_client #creates a connection to Supabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings #generates vector embeddings
from langchain_core.tools import tool #decorator to create LangChain tools that AI agents can use
from typing import Optional, Tuple, Union #type hints for better code docs
from functools import wraps #decorator to preserve function metadata when creating decorators

# Configure logging
# Creates a logger instance named after the current module
logger = logging.getLogger(__name__)

# Custom Exceptions for better error handling
class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class DatabaseError(ToolError):
    """Exception for database-related errors."""
    pass

class ValidationError(ToolError):
    """Exception for input validation errors."""
    pass

# Configuration class storing constants used throughout the module
class ToolConfig:
    """Configuration constants for tools."""
    MATCH_THRESHOLD = 0.5   #minimum similarity score for document matching
    MATCH_COUNT = 10        #maximum number of documents to return
    DEFAULT_VAT_RATE = 8    #default VAT percentage
    MIN_QUERY_LENGTH = 2    #minimum search query length

# Singleton Database Client for managing database connections
class DatabaseClient:
    """Thread-safe singleton client for database operations."""
    _instance = None        #class variable to store the single instance
    _initialized = False    #flag to track if initialization has been done

    # The __new__ method controls instance creation
    def __new__(cls):
        if cls._instance is None: #if no instance exists, 
            cls._instance = super().__new__(cls) #create one using the parent class's __new__
        return cls._instance #always return the existing instance
    
    # The __init__ method
    def __init__(self):
        if not self._initialized: #checks if initialization has been done
            self._initialize()    #calls _initialize() only once
            DatabaseClient._initialized = True  #sets the _initialized flag to True
    
    # Loads environment variables from .env file
    def _initialize(self):
        """Initialize database connection and embeddings."""
        load_dotenv()

        # Initialize Supabase
        # Gets Supabase URL and key from environment variables
        supabase_url = os.getenv("SUPABASE_URL") #get URL from .env
        supabase_key = os.getenv("SUPABASE_KEY") #get key from .env
        # Raise an error if either is missing
        if not supabase_url or not supabase_key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        self.supabase = create_client(supabase_url, supabase_key) #create the client instance

        # Initialize embeddings
        gemini_api_key = os.getenv("GEMINI_API_KEY") #get API key from environment
        if not gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment") # raise error if missing
        #initialize the embeddings model with specific settings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )

    # Getter methods to access the Supabase client and embeddings model
    def get_client(self):
        """Get the Supabase client."""
        return self.supabase

    def get_embeddings(self):
        """Get the embeddings model."""
        return self.embeddings

# Helper Functions to convert numeric inputs to float
def parse_numeric_input(value: Union[str, float, int]) -> float:
    """Parse and clean numeric input from various formats."""
    if isinstance(value, (int, float)):
        return float(value)
    
    # Clean string input
    cleaned = str(value).strip()    #removes whitespace
    cleaned = cleaned.replace(",", "").replace("_", "") #removes commas and underscores
    cleaned = cleaned.replace("k", "000").replace("K", "000")   #replaces 'k' or 'K' with '000'
    cleaned = cleaned.replace("%", "")  #removes percent signs

    # Attempts to convert cleaned string to float, raises ValidationError on failure
    try:
        return float(cleaned)
    except ValueError:
        raise ValidationError(f"Cannot parse numeric value: {value}")

# Formats a document for display
def format_document_result(doc: dict) -> str:
    """Format a single document result for AI consumption."""
    meta = doc.get('metadata') or {}    #gets metadata or empty dict
    domain = meta.get('domain', 'GENERAL') #extracts domain
    priority = meta.get('priority_label', 'LEGACY') #extracts priority with defaults

    # Returns formatted string with document information
    return (
        f"[{domain}] [{priority}]\n"
        f"DOCUMENT: {meta.get('official_name', 'Unknown')}\n"
        f"EFFECTIVE: {meta.get('effective_date', 'Unknown')}\n"
        f"CONTENT: {doc.get('content', '')}\n"
        "--------------------------------------------------"
    )

# Decorator to validate search queries
def validate_search_query(func):
    """Decorator to validate search query input."""
    @wraps(func)    #uses wraps to preserve function metadata
    def wrapper(*args, **kwargs):   #extracts query from either positional or keyword arguments
        # The query should be the first argument
        if args:
            query = args[0]
        elif 'query' in kwargs:
            query = kwargs['query']
        else:
            raise ValidationError("Search query not found in arguments") #raises error if query not found
        
        if not query or not query.strip():  #checks if query if not empty
            raise ValidationError("Search query cannot be empty")
        if len(query.strip()) < ToolConfig.MIN_QUERY_LENGTH:    #checks minimum length
            raise ValidationError(f"Search query too short (minimum {ToolConfig.MIN_QUERY_LENGTH} characters)")
        
        # Update the query in kwargs
        if 'query' in kwargs:
            kwargs['query'] = query.strip() #strips whitespace from query
            return func(**kwargs)
        else:
            return func(query.strip(), *args[1:])
    return wrapper  #calls the original function with cleaned query

# Database Operations
def search_documents(query: str, db_client: DatabaseClient) -> list:
    """Search documents in the database"""
    try:
        embeddings = db_client.get_embeddings()     #gets embeddings model
        query_vector = embeddings.embed_query(query)    #generates vector embedding for the query

        supabase = db_client.get_client()   #gets supabase client
        response = supabase.rpc(            #calls remote procedure named "match_documents" with params
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_threshold": ToolConfig.MATCH_THRESHOLD,
                "match_count": ToolConfig.MATCH_COUNT
            }
        ).execute() #executes the call

        # Supabase returns data in response.data
        if hasattr(response, 'data'):
            return response.data    #returns data from response if available
        else:
            # Try to access it directly
            return []               #returns empty list otherwise
    except Exception as e:
        logger.error(f"Database search error: {e}") #logs and re-raised errors as DatabaseError
        raise DatabaseError(f"Failed to search documents: {str(e)}")

def format_search_results(documents: list) -> str:
    """Format search results for AI consumption."""
    if not documents:   #returns message if no documents found
        return "SYSTEM: No relevant documents found. Do not answer from memory."

    formatted_results = []
    for doc in documents:
        formatted_results.append(format_document_result(doc))

    return "\n".join(formatted_results) #formats and joins them with newlines

# Tool Definitions
@tool   #decorator to make it usable by LangChain agents
@validate_search_query  #decorator to validate input
def search_accounting_law(query: str):
    """
    Use this tool to find information about Vietnamese Accounting Laws, 
    Circulars (like Circular 200), Tax regulations, or specific Accounts (TK 111, etc).
    Input should be a search query like "Definition of Account 111".
    """
    logger.info(f"Agent is searching database for: {query}")    #logs the search query

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

    # Handles different types of errors with appropriate logging and user messages
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return f"Validation error: {str(e)}"
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return f"Database error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in search_accounting_law: {e}")
        return f"Unexpected error: {str(e)}"

@tool #decorator
def calculate_vat(amount: Union[str, float, int], rate_percent: Union[str, float, int] = ToolConfig.DEFAULT_VAT_RATE):
    """
    Use this tool to calculate VAT (Value Added Tax) or generic percentages.
    Input: amount (number), rate_percent (number, default is 8).
    """
    try:
        # Parse inputs
        f_amount = parse_numeric_input(amount)
        f_rate = parse_numeric_input(rate_percent)
        
        # Validate inputs: amount and rate
        if f_amount < 0:
            raise ValidationError("Amount cannot be negative")
        if f_rate < 0 or f_rate > 100:
            raise ValidationError("Rate must be between 0 and 100 percent")
        
        logger.info(f"Agent is calculating VAT: {f_amount} * {f_rate}%") #logs calculation
        
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
