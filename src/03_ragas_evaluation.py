"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Giữ GIỐNG HỆT SYSTEM_V1 / SYSTEM_V2 ở 02_prompt_hub_ab_routing.py để kết quả so sánh được.
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # list[str] cho RAGAS

    # Chuỗi ghép chỉ dùng để điền {context} trong prompt
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


DATA_DIR = Path(__file__).parent.parent / "data"


def load_or_collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Lưu kết quả RAG ra data/rag_outputs_<version>.json và dùng lại ở lần chạy sau.
    RAGAS chạy lâu và dễ dính rate limit — nếu phải chạy lại thì không cần sinh lại 50 câu trả lời.
    Xóa file JSON để bắt buộc sinh lại.
    """
    cache_path = DATA_DIR / f"rag_outputs_{prompt_version}.json"
    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(results) == len(QA_PAIRS):
            print(f"\n♻️  Dùng lại {len(results)} kết quả RAG {prompt_version} từ {cache_path.name}")
            return results

    results = collect_rag_outputs(vectorstore, prompt_version)
    cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu kết quả RAG {prompt_version} vào {cache_path.name}")
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def make_judge(provider: str, model_name: str, run_config: RunConfig) -> LangchainLLMWrapper:
    """
    Tạo LLM judge cho RAGAS (provider/model rỗng → dùng giá trị mặc định trong .env).

    bypass_n: answer_relevancy xin n=3 câu trả lời trong 1 request; API OpenAI-compatible
    như Groq chỉ cho n=1 → gọi 3 request riêng thay vì 1 request với n=3.
    """
    llm = get_llm(provider=provider or None, temperature=0, model=model_name or None)
    return LangchainLLMWrapper(llm, run_config=run_config, bypass_n=True)


def assign_metric_judges(metrics: list, run_config: RunConfig):
    """
    Gắn judge riêng cho metric nào được cấu hình RAGAS_JUDGE_<METRIC>=provider:model.
    Metric không cấu hình → evaluate() dùng judge mặc định (tham số llm=).
    Mỗi metric luôn dùng cùng 1 judge cho cả V1 và V2 nên so sánh V1 vs V2 vẫn công bằng.
    """
    for metric in metrics:
        spec = config.RAGAS_METRIC_JUDGES.get(metric.name)
        if spec:
            provider, model_name = spec.split(":", 1)
            metric.llm = make_judge(provider, model_name, run_config)
        judge = spec or f"{config.RAGAS_JUDGE_PROVIDER or config.PROVIDER}:{config.RAGAS_JUDGE_MODEL or 'default'}"
        print(f"  ⚖️  {metric.name:20s} ← {judge}")


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # Free tier giới hạn token/phút → timeout + retry dài hơn mặc định
    run_config = RunConfig(max_workers=8, timeout=900, max_retries=15, max_wait=90)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = make_judge(config.RAGAS_JUDGE_PROVIDER, config.RAGAS_JUDGE_MODEL, run_config)
    emb_eval = get_embeddings()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    assign_metric_judges(metrics, run_config)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    # Bỏ qua None/NaN (sample bị lỗi parse hoặc timeout khi chấm) để không kéo cả mean thành NaN
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw   = result[key]
        valid = [v for v in raw if v is not None and not np.isnan(v)]
        if len(valid) < len(raw):
            print(f"  ℹ️  {key}: bỏ qua {len(raw) - len(valid)} sample không có điểm")
        scores[key] = float(np.mean(valid)) if valid else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = load_or_collect_rag_outputs(vectorstore, "v1")
    v2_results = load_or_collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "num_qa_pairs": {"v1": len(v1_results), "v2": len(v2_results)},
        "provider": config.PROVIDER,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
