import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file or environment.")

# Ensure compatibility with libraries expecting GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Base directories scoped to this subproject
DOCS_DIR = str(BASE_DIR / "docs")
UPLOADS_DIR = str(BASE_DIR / "uploads")
VECTOR_STORE_DIR = str(BASE_DIR / "vector_store")

# Models
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSION = 768
GENERATION_MODEL = "gemini-2.5-pro"


