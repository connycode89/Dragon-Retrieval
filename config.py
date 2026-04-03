import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DUCHAS_API_KEY = os.getenv("DUCHAS_API_KEY", "")
LOGAINM_API_KEY = os.getenv("LOGAINM_API_KEY", "")

# Paths
PROJECT_ROOT = Path(__file__).parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# API base URLs
DUCHAS_API_BASE = "https://www.duchas.ie/api/v0.6"
LOGAINM_API_BASE = "https://www.logainm.ie/api/v1.0"
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"

# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 6

# Irish county IDs for Dúchas API (subset of most folklore-rich counties)
# Full list available at: https://www.duchas.ie/api/v0.6/counties
DUCHAS_COUNTY_IDS = {
    "Clare": 10,
    "Cork": 14,
    "Donegal": 16,
    "Galway": 22,
    "Kerry": 27,
    "Leitrim": 32,
    "Mayo": 39,
    "Roscommon": 49,
    "Sligo": 52,
    "Tipperary": 54,
}
