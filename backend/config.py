"""Central settings. Every other module imports its constants from here."""
import os
from pathlib import Path

# ---- Folders -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"          # local LLM / embedding models live here
LOGS_DIR = BASE_DIR / "logs"              # saved PDFs + Q&A csv
PDF_LOG_DIR = LOGS_DIR / "pdfs"
CSV_LOG_PATH = LOGS_DIR / "qa_log.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

for _folder in (MODELS_DIR, LOGS_DIR, PDF_LOG_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ---- Server --------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000

# ---- Models (served locally by Ollama) -----------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")      # answers questions
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")      # multilingual (Persian + English)
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 300

# ---- PDF handling --------------------------------------------------------
MAX_UPLOAD_MB = 50
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # characters shared between neighbouring chunks
TOP_K = 4                # chunks sent to the LLM with each question
EMBED_BATCH_SIZE = 16

# ---- OCR (scanned / image-only pages) -------------------------------------
OCR_LANGUAGES = ["fa", "en"]     # Persian + English
OCR_DPI = 200                    # page render quality (higher = slower, more accurate)
OCR_MAX_PAGES = 30               # refuse to OCR more pages than this in one PDF
OCR_MIN_IMAGE_COVERAGE = 0.3     # a page is "scanned" if pictures cover >= 30% of it
OCR_MIN_TEXT_CHARS = 30          # ...and it has fewer characters of real text than this
OCR_USE_GPU = False
