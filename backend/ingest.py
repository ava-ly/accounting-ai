import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load Secrets
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not url or not key:
    raise ValueError("Supabase credentials missing in .env")

# Setup Clients
supabase: Client = create_client(url, key)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=gemini_key
)

def ingest_file(
    file_path: str, 
    official_name: str,
    effective_date: str,        # Format: "YYYY-MM-DD"
    priority_label: str = "LEGACY",
    domain: str = "VAS"
    ):

    print(f"Reading {file_path} [{priority_label}]...")

    # Determine the Display Name
    filename = os.path.basename(file_path)
    display_name = official_name if official_name else filename
    print(f"Tagging as: {display_name}")

    # Load PDF
    if file_path.endswith(".docx"):
        print("Detected Word Document (.docx)")
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".pdf"):
        print("Detected PDF")
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .docx or .pdf")
    
    pages = loader.load()
    print(f"Loaded {len(pages)} pages/sections.")

    # Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Split by "Điều" followed by a number
        separators=[
            "\nĐiều ",      # Best split point (Vietnamese)
            "\nArticle ",   # Best split point (English)
            "\nPhần ",      # Part
            "\nChương ",    # Chapter
            "\n\n",         # Paragraphs
            "\n",           # Lines
            " "             # Words
        ],
        chunk_size=3000, # Make chunks bigger to keep Articles together
        chunk_overlap=200,
        keep_separator=True
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and Save to Supabase
    print(f"Generating Vectors and Uploading... (This may take a moment)")

    for i, chunk in enumerate(chunks):
        # Turn Text into Numbers
        # In a real PRODUCTION app, we'd batch this. For learning, one by one is fine/clearer.
        vector = embeddings.embed_query(chunk.page_content)

        # Prepare Data
        page_num = chunk.metadata.get("page", 0)
        if page_num == 0:
            page_num = "Text Section"
        
        data = {
            "content": chunk.page_content,
            "metadata": {
                "source": filename,
                "official_name": display_name,
                "effective_date": effective_date,
                "priority_label": priority_label,
                "domain": domain,
                "page": page_num
            },
            "embedding": vector
        }

        # Insert
        supabase.table("documents").insert(data).execute()

        if i % 10 == 0:
            print(f"Saved chunk {i}/{len(chunks)}")

    print("Ingestion Complete!")

if __name__ == "__main__":
    # # New Circular 99
    # ingest_file(
    #     "data/circular_99_2025.docx", 
    #     official_name="Thông tư 99/2025/TT-BTC",
    #     effective_date="2026-01-01",
    #     priority_label="LATEST",
    #     domain="VAS"
    # )

    # # Old Circular 200
    # ingest_file(
    #     "data/circular_200.docx", 
    #     official_name="Thông tư 200/2014/TT-BTC",
    #     effective_date="2015-01-01",
    #     priority_label="LEGACY",
    #     domain="VAS"
    # )
    
    # # Tax law - VAT
    # ingest_file(
    #     "data/circular_69_2025_vat.docx", 
    #     official_name="Thông tư 69/2025/TT-BTC",
    #     effective_date="2025-07-01",
    #     priority_label="LATEST",
    #     domain="VAT"
    # )

    # Tax law - CIT
    ingest_file(
        "data/decree_320_2025_cit.docx", 
        official_name="Nghị định 320/2025/NĐ-CP",
        effective_date="2025-01-01",
        priority_label="LATEST",
        domain="CIT"
    )