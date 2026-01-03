import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from typing import Optional, Tuple, Union

# Setup
_env_loaded = False
_supabase = None
_embeddings = None


def _ensure_initialized() -> Tuple[object, GoogleGenerativeAIEmbeddings]:
    global _env_loaded, _supabase, _embeddings

    if not _env_loaded:
        load_dotenv()
        _env_loaded = True

    if _supabase is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        _supabase = create_client(supabase_url, supabase_key)

    if _embeddings is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )

    return _supabase, _embeddings

# Define the tool
@tool
def search_accounting_law(query: str):
    """
    Use this tool to find information about Vietnamese Accounting Laws, 
    Circulars (like Circular 200), Tax regulations, or specific Accounts (TK 111, etc).
    Input should be a search query like "Definition of Account 111".
    """
    print(f"Agent is searching database for: {query}")

    try:
        supabase, embeddings = _ensure_initialized()

        # Generate vector
        query_vector = embeddings.embed_query(query)

        # Search Supabase
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.5,
                "match_count": 10
            }
        ).execute()

        # Format results for AI to read
        if not response.data:
            print("Database returned 0 matches (below threshold).")
            return "SYSTEM: No relevant documents found. Do not answer from memory."

        # Format for Gemini
        formatted_results = []
        for d in response.data:
            meta = d.get('metadata') or {}

            # Check Priority
            domain = meta.get('domain', 'GENERAL')
            priority = meta.get('priority_label', 'LEGACY')

            # # Visual Marker for the AI
            # domain_tag = f"[{domain}]"
            # status_line = ""
            # if label == "LATEST":
            #     status_line = "STATUS: PRIMARY AUTHORITY (LATEST LAW)"
            # else:
            #     status_line = "STATUS: REFERENCE ONLY (OLD LAW)"
            
            entry = (
                f"[{domain}] [{priority}]\n"
                f"DOCUMENT: {meta.get('official_name')}\n"
                f"EFFECTIVE: {meta.get('effective_date')}\n"
                f"CONTENT: {d.get('content')}\n"
                "--------------------------------------------------"
            )
            formatted_results.append(entry)
            
        print(f"Selected Top {len(formatted_results)} highest quality docs.")
        return "\n".join(formatted_results)

    except Exception as e:
        print(f"Error: {e}")
        return f"Error searching database: {str(e)}"

@tool
def calculate_vat(amount: Union[str, float, int], rate_percent: Union[str, float, int] = 8):
    """
    Use this tool to calculate VAT (Value Added Tax) or generic percentages.
    Input: amount (number), rate_percent (number, default is 8).
    """
    try:
        # 1. Clean the input (remove commas, 'k', etc)
        # Llama might send "100,000" or "100k"
        s_amount = str(amount).replace(",", "").replace("_", "")
        s_amount = s_amount.replace("k", "000").replace("K", "000")
        s_rate = str(rate_percent).replace("%", "")

        # 2. Convert to float
        f_amount = float(s_amount)
        f_rate = float(s_rate)

        print(f"Agent is calculating VAT: {f_amount} * {f_rate}%")
        
        tax = f_amount * (f_rate / 100)
        total = f_amount + tax
        return f"Amount: {f_amount:,.0f} | Tax ({f_rate}%): {tax:,.0f} | Total: {total:,.0f}"

    except ValueError:
        return "Error: Please provide valid numbers for calculation."
    except Exception as e:
        return f"Error: {str(e)}"
