# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Bùi Thị Thu Uyên — 2A202602613
**LangSmith project:** `day22-buithithuuyen`
**Prompt Hub:** `bui-thi-thu-uyen-rag-prompt-v1`, `bui-thi-thu-uyen-rag-prompt-v2`

## Danh sách file

| File | Nội dung |
|---|---|
| `01_langsmith_traces.png` | 50 trace `rag-query` của Bước 1 (filter `name:"rag-query"`) |
| `01_rag_pipeline_log.txt` | Log console Bước 1 (50 Q/A) |
| `02_prompt_hub.png` | 2 prompt V1, V2 trên LangSmith Prompt Hub |
| `02_ab_trace_detail.png` | 1 trace `ab-rag-query`: retriever (k=3) → ChatOpenAI, output có `version` |
| `02_ab_routing_log.txt` | Log A/B routing: mỗi câu có `request_id` + nhãn `[prompt-v1]` / `[prompt-v2]` |
| `03_ragas_scores.png` | Bảng so sánh RAGAS V1 vs V2 trên terminal |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json` |
| `03_ragas_log.txt` | Log console Bước 3 |
| `04_pii_demo_log.txt` | 7 test case PIIDetector |
| `04_json_demo_log.txt` | 6 test case JSONFormatter |

## Môi trường chạy

Bài chạy hoàn toàn bằng tài nguyên miễn phí:

| Thành phần | Dùng | Lý do |
|---|---|---|
| LLM sinh câu trả lời (Bước 1–3) | Groq `qwen/qwen3.8-27b` qua API OpenAI-compatible (`PROVIDER=openai` + `OPENAI_BASE_URL`) | Gemini free bị lỗi 402 "prepayment credits depleted", OpenAI key hết credit |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` chạy local (`EMBEDDING_PROVIDER=local`) | Groq không có embeddings API |
| RAGAS judge | `faithfulness`, `context_precision` → Ollama `qwen2.5:3b` (GPU local); `answer_relevancy` → Groq `gpt-oss-20b`; `context_recall` → Groq `gpt-oss-120b` | Groq free chỉ cho 200k token/ngày mỗi model, RAGAS cần khoảng 800k token. Phải chia metric cho nhiều judge |

**Mỗi metric dùng cùng một judge cho cả V1 và V2**, nên so sánh V1 với V2 trên từng metric vẫn công bằng. Cấu hình nằm trong `.env.example`: `RAGAS_JUDGE_PROVIDER`, `RAGAS_JUDGE_MODEL`, `RAGAS_JUDGE_<METRIC>`.

## A/B routing

`request_id = req-0000 … req-0049` → `int(md5(request_id), 16) % 2`: chẵn đi V1, lẻ đi V2.
- **Kết quả:** V1 = 19 câu, V2 = 31 câu.
- MD5 chia đều về lâu dài; với chỉ 50 ID thì lệch 19/31 là dao động ngẫu nhiên bình thường.
- **Tính tất định:** route mỗi ID 2 lần đều ra cùng version (log in `🔒 Routing tất định: 50/50`), và chạy lại script cho đúng cùng phân bổ.

## Kết quả RAGAS (50 cặp QA × 2 phiên bản)

| Metric | V1 (ngắn gọn) | V2 (chuyên gia, có cấu trúc) | Chênh lệch |
|---|---|---|---|
| faithfulness | 0.9057 | **0.9066** | +0.001 (V2) |
| answer_relevancy | **0.9343** | 0.9141 | +0.020 (V1) |
| context_recall | 1.0000 | 1.0000 | 0 |
| context_precision | 0.9617 | 0.9617 | 0 |

Faithfulness ≥ 0.9 ở **cả 2** phiên bản. Không có sample nào bị NaN (400/400 điểm hợp lệ).

Đặc điểm câu trả lời (tính từ `data/rag_outputs_v*.json`):

| | V1 | V2 |
|---|---|---|
| Số từ trung bình | 45 | 70 |
| Số câu trung bình | 2.2 | 3.3 |

Cả hai prompt đều được model tuân thủ đúng độ dài yêu cầu (V1: 2–4 câu, V2: 3–5 câu).

## Phân tích: vì sao V1 và V2 khác (và giống) nhau

**1. `context_recall` và `context_precision` bằng nhau tuyệt đối, đúng như kỳ vọng.**
- Hai metric này chỉ chấm **phần retrieval**: so `retrieved_contexts` với `reference` (và câu hỏi), không nhìn vào câu trả lời.
- V1 và V2 dùng chung FAISS index, chung `k=3`, chung câu hỏi, nên nhận về cùng 3 chunk. Prompt khác nhau không ảnh hưởng đến 2 điểm này.
- **Muốn cải thiện chúng phải sửa retriever** (chunk_size, k, hybrid search, reranking), không phải sửa prompt.
- `context_recall = 1.0` cho thấy với bộ 50 câu hỏi này, top-3 chunk luôn chứa đủ thông tin của đáp án chuẩn. `context_precision ≈ 0.96` nghĩa là thỉnh thoảng có 1 chunk ít liên quan bị xếp trên chunk hữu ích.

**2. `answer_relevancy`: V1 cao hơn V2 khoảng 0.02.**
- RAGAS tính metric này bằng cách cho LLM sinh ngược câu hỏi từ câu trả lời, rồi đo cosine similarity với câu hỏi gốc.
- **V1** trả lời thẳng vào ý chính trong 2–4 câu, nên câu hỏi sinh ngược bám sát câu hỏi gốc.
- **V2** được yêu cầu bổ sung "định nghĩa, thành phần, cơ chế hoặc ví dụ". Phần mở rộng đó kéo câu hỏi sinh ngược sang hướng rộng hơn, ví dụ hỏi thêm về Adam/RMSprop khi câu hỏi gốc chỉ là "What is backpropagation?". Vì vậy similarity giảm nhẹ.
- Đây là **đánh đổi giữa độ sâu và độ tập trung**: câu trả lời đầy đủ hơn không đồng nghĩa với liên quan hơn theo cách RAGAS đo.

**3. `faithfulness`: gần như bằng nhau (V2 nhỉnh 0.001).**
- Cả hai prompt đều ép "chỉ dùng Context". V2 còn ghi rõ "Mọi nhận định phải được Context hỗ trợ".
- V2 dài hơn khoảng 55%, tức nhiều claim hơn, nhưng **không** làm giảm tỉ lệ claim có căn cứ. Câu dặn chỉ được nói điều có trong Context đã bù được rủi ro mà câu trả lời dài hơn mang lại.
- Chênh lệch 0.001 trên 50 mẫu là quá nhỏ để kết luận V2 trung thực hơn. Kết luận đúng là **hai prompt trung thực ngang nhau**.

**Kết luận chọn prompt:**
- **V1** hợp với hỏi đáp nhanh, nơi độ tập trung quan trọng: relevancy cao hơn, ngắn hơn nên rẻ token hơn.
- **V2** hợp khi người dùng cần giải thích sâu. V2 thêm chi tiết mà không mất faithfulness, chỉ giảm nhẹ relevancy.

## Hạn chế
- Judge khác nhau giữa các metric do giới hạn free tier, nên **không** so sánh trực tiếp giá trị giữa các metric với nhau. So sánh V1 với V2 trong cùng một metric vẫn hợp lệ.
- `qwen2.5:3b` là judge nhỏ. Điểm faithfulness tuyệt đối có thể lệch so với judge lớn như GPT-4o, nhưng sai lệch đó áp dụng như nhau cho V1 và V2.
- Chỉ 50 câu hỏi, 1 lần chạy (temperature = 0). Các chênh lệch dưới khoảng 0.01 nên xem là hòa.
- Ở dòng "Winner" của bảng terminal, logic có sẵn trong template in `← V2` khi hai điểm bằng nhau (context_recall 1.0 = 1.0). Thực tế là hòa.
