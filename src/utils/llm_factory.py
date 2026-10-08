"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class LocalEmbeddings(Embeddings):
    """
    Embeddings chạy trên máy bằng FastEmbed (ONNX, CPU) — miễn phí, không cần API key.

    Bọc FastEmbedEmbeddings vì RAGAS ghi log `embeddings.model` và yêu cầu kiểu str,
    trong khi FastEmbedEmbeddings.model là object TextEmbedding → ValidationError.
    """

    def __init__(self, model_name: str):
        from langchain_community.embeddings import FastEmbedEmbeddings
        self.model  = model_name
        self._inner = FastEmbedEmbeddings(model_name=model_name)   # tải model 1 lần rồi cache

    def embed_documents(self, texts: list) -> list:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list:
        return self._inner.embed_query(text)


def get_llm(provider: str = None, temperature: float = 0.0, model: str = None):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)
        model       : ghi đè tên model (provider "openai" hoặc "ollama"),
                      vd dùng model khác làm RAGAS judge

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = model or config.OPENAI_MODEL
        kwargs = {
            "model": model_name,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
            "max_retries": config.LLM_MAX_RETRIES,   # client tự chờ theo retry-after khi gặp 429
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.LLM_MAX_TOKENS:
            kwargs["max_tokens"] = config.LLM_MAX_TOKENS
        if "gpt-oss" in model_name:
            # gpt-oss là reasoning model: "low" giữ token suy luận ít → đỡ tốn quota token/phút
            kwargs["reasoning_effort"] = "low"
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=config.OLLAMA_NUM_CTX,       # prompt RAGAS có few-shot dài ~1.5k token
            num_predict=config.LLM_MAX_TOKENS,   # chặn output chạy quá dài
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter" | "local"
                  Mặc định: đọc EMBEDDING_PROVIDER từ .env (fallback về PROVIDER)

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()

    if provider == "local":
        return LocalEmbeddings(config.LOCAL_EMBEDDING_MODEL)

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
