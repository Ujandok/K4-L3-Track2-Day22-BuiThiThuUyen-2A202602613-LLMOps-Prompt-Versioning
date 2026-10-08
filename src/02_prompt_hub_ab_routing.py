"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "bui-thi-thu-uyen-rag-prompt-v1"
PROMPT_V2_NAME = "bui-thi-thu-uyen-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — "trả lời nhanh": ngắn gọn 2-4 câu, đi thẳng vào ý chính.
# ⚠️ BẮT BUỘC có {context} trong SYSTEM — thiếu thì LLM không nhận được tài liệu
#    mà chương trình KHÔNG báo lỗi (câu trả lời bịa, điểm RAGAS thấp).
SYSTEM_V1 = (
    "Bạn là trợ lý hỏi đáp kỹ thuật về AI/ML. Trả lời NGẮN GỌN trong 2-4 câu, "
    "đi thẳng vào ý chính của câu hỏi.\n"
    "- Chỉ dùng thông tin có trong Context bên dưới, không thêm kiến thức bên ngoài.\n"
    "- Trả lời bằng cùng ngôn ngữ với câu hỏi.\n"
    "- Nếu Context không có thông tin cần thiết, nói rõ là tài liệu không đề cập.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2 — "chuyên gia phân tích": tự xác định facts trước, rồi trình bày có cấu trúc 3-5 câu.
# ⚠️ BẮT BUỘC có {context} (giống SYSTEM_V1)
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích tài liệu AI/ML. Trước khi trả lời, hãy âm thầm "
    "xác định các facts trong Context liên quan trực tiếp đến câu hỏi.\n"
    "Sau đó viết câu trả lời có tổ chức gồm 3-5 câu: câu đầu nêu ý chính, các câu sau "
    "bổ sung chi tiết (định nghĩa, thành phần, cơ chế hoặc ví dụ) lấy từ Context.\n"
    "- Mọi nhận định phải được Context hỗ trợ; không suy đoán ngoài Context.\n"
    "- Nếu Context chỉ trả lời được một phần, nêu rõ phần nào không có trong tài liệu.\n"
    "- Trả lời bằng cùng ngôn ngữ với câu hỏi; chỉ đưa ra câu trả lời cuối cùng.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def _push_one(client: Client, name: str, template: ChatPromptTemplate, description: str, label: str):
    """
    Push 1 prompt lên Hub. Lỗi 409 "Nothing to commit" nghĩa là nội dung không đổi
    so với commit mới nhất trên Hub → không phải lỗi, Hub giữ nguyên phiên bản cũ.
    """
    try:
        url = client.push_prompt(name, object=template, description=description)
        print(f"✅ Đã push {label} → {url}")
    except Exception as e:
        if "409" in str(e) or "Nothing to commit" in str(e):
            print(f"ℹ️  {label} không đổi so với Hub (409 Nothing to commit) — giữ phiên bản hiện tại")
        else:
            print(f"⚠️  {label} lỗi: {e}")


def push_prompts_to_hub(client: Client):
    """Upload cả 2 prompt templates lên LangSmith Prompt Hub."""
    _push_one(client, PROMPT_V1_NAME, PROMPT_V1, "V1 – ngắn gọn 2-4 câu", "V1")
    _push_one(client, PROMPT_V2_NAME, PROMPT_V2, "V2 – chuyên gia, có cấu trúc 3-5 câu", "V2")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}' ({e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}' ({e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic) — MD5 không
    phụ thuộc vào process/seed, khác với hash() built-in của Python hay random.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def check_routing_is_deterministic(request_ids: list) -> bool:
    """Route mỗi request_id 2 lần và xác nhận luôn ra cùng version."""
    return all(get_prompt_version(rid) == get_prompt_version(rid) for rid in request_ids)


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs    = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    """Tạo FAISS vectorstore từ knowledge base (giống Bước 1)."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    request_ids = [f"req-{i:04d}" for i in range(len(SAMPLE_QUESTIONS))]
    if check_routing_is_deterministic(request_ids):
        print(f"🔒 Routing tất định: {len(request_ids)}/{len(request_ids)} request_id cho cùng version qua 2 lần route")

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = request_ids[i]

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        # langsmith_extra gắn version + request_id vào trace để lọc V1/V2 trên dashboard
        result = ask_ab(
            retriever, llm, prompt, question, version_tag,
            langsmith_extra={
                "tags": [f"prompt-{version_tag}"],
                "metadata": {"request_id": request_id, "prompt_name": version_key},
            },
        )

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        short_answer = " ".join(result["answer"].split())[:100]
        print(f"[{i+1:02d}] {request_id} [prompt-{version_tag}] {question[:55]}...")
        print(f"      A: {short_answer}")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
