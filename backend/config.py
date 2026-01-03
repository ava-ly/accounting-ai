# Centralizes all configuration so you can switch models or update without hunting through code.

"""Configuration module for the accounting AI agent."""
import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_accounting_law, calculate_vat

# Load environment variables
load_dotenv()

class Config:
    """Configuration constants and settings."""
    
    # Model configuration
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    TEMPERATURE = 0
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Tools
    TOOLS: List = [search_accounting_law, calculate_vat]
    
    # Logging
    ENABLE_DEBUG_PRINTING = True
    
    @classmethod
    def validate(cls) -> None:
        """Validate required environment variables."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")

def create_llm() -> ChatGroq:
    """Create and configure the LLM instance."""
    Config.validate()
    return ChatGroq(
        model=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE,
        api_key=Config.GROQ_API_KEY
    ).bind_tools(Config.TOOLS)