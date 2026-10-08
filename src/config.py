"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Provider cho embeddings — mặc định giống PROVIDER.
# "local": chạy FastEmbed trên CPU, miễn phí, không cần API key
#          (dùng khi LLM là API OpenAI-compatible không có endpoint embeddings, vd Groq/Cerebras).
EMBEDDING_PROVIDER    = os.getenv("EMBEDDING_PROVIDER", PROVIDER).lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MAX_RETRIES        = int(os.getenv("LLM_MAX_RETRIES", "2"))   # tăng khi dùng free tier hay bị 429
# Giới hạn token output mỗi request. Groq free có hạn mức output-token/phút (OTPM);
# không đặt max_tokens thì Groq ước lượng output rất lớn và từ chối request ("Request too large").
LLM_MAX_TOKENS         = int(os.getenv("LLM_MAX_TOKENS", "0")) or None

# ── RAGAS judge (tùy chọn) ────────────────────────────────────────────────
# Để trống → RAGAS dùng cùng provider/model với RAG.
RAGAS_JUDGE_PROVIDER = os.getenv("RAGAS_JUDGE_PROVIDER", "").lower()   # vd "ollama"
RAGAS_JUDGE_MODEL    = os.getenv("RAGAS_JUDGE_MODEL", "")
# Ghi đè judge cho từng metric bằng "provider:model", vd RAGAS_JUDGE_CONTEXT_RECALL=openai:openai/gpt-oss-120b
# (free tier giới hạn token/ngày theo từng model → chia metric cho nhiều model chạy song song)
RAGAS_METRIC_JUDGES = {
    name: os.getenv(f"RAGAS_JUDGE_{name.upper()}", "")
    for name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
}

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX          = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []

    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY (LangSmith)")

    if PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    elif PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    # Ollama: không cần API key

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  Provider: {PROVIDER.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    validate()
