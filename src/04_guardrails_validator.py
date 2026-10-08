"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 7 trường hợp PII và 6 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import re
import json

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # Regex patterns cho từng loại PII.
    # PHONE: pattern gốc bắt đầu bằng \b nên không khớp được dấu "(" (ký tự không phải word)
    #        → "(555) 867-5309" bị redact thành "([PHONE_REDACTED]". Tách 2 nhánh:
    #        "(555) " có ngoặc, hoặc "555-" không ngoặc (nhánh này giữ \b).
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về FailResult với fix_value là text đã xử lý.

        ⚠️ Với OnFailAction.FIX, Guardrails CHỈ thay output bằng FailResult.fix_value.
           PassResult(value_override=...) KHÔNG có tác dụng → output giống hệt input.

        Bước:
          1. Copy value → redacted_text
          2. Với mỗi loại PII và pattern tương ứng:
             - Tìm tất cả matches bằng re.findall(pattern, value)
             - Thay thế từng match bằng "[PII_TYPE_REDACTED]" trong redacted_text
             - Ghi lại (pii_type, match) vào found_pii
          3. Nếu found_pii không rỗng → FailResult(error_message=..., fix_value=redacted_text)
          4. Nếu không tìm thấy PII → PassResult()
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)

            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            # FIX chỉ dùng fix_value của FailResult để thay output
            return FailResult(error_message="Phát hiện PII", fix_value=redacted_text)

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.

        Bước:
          1. Strip whitespace đầu/cuối
          2. Xóa markdown fences bằng re.sub
          3. Thay single quotes → double quotes
          4. Xóa trailing commas trước } hoặc ]
          5. Trả về chuỗi đã sửa (chưa re-serialize)
        """
        text = text.strip()

        # Xóa markdown fences — đã cho sẵn
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Nháy đơn → nháy đôi (JSON chỉ chấp nhận nháy đôi)
        text = text.replace("'", '"')

        # Xóa dấu phẩy thừa ngay trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON.
        Nếu thất bại, gọi _repair() rồi thử lại.

        - JSON hợp lệ sẵn          → PassResult()
        - Sửa được                 → FailResult(error_message=..., fix_value=json.dumps(parsed, indent=2))
        - Không sửa được           → FailResult(error_message=..., fix_value=<JSON dự phòng>)

        ⚠️ Với OnFailAction.FIX, chỉ FailResult.fix_value mới thay được output.
        """
        # 1) JSON hợp lệ sẵn → giữ nguyên
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # 2) Sửa rồi parse lại → trả JSON đã chuẩn hóa qua fix_value
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print(f"  🔧 JSON đã được sửa thành công")
            return FailResult(error_message="JSON lỗi, đã tự sửa", fix_value=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            # Không sửa được → trả về JSON dự phòng để output vẫn là JSON hợp lệ
            fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]}, ensure_ascii=False)
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


def describe_outcome(result) -> str:
    """
    Mô tả kết quả của guard.validate().

    Với OnFailAction.FIX, validation_passed = True cả khi output đã bị sửa,
    nên phải đọc validation_summaries để biết validator có trả FailResult hay không.
    """
    failures = [s for s in (result.validation_summaries or []) if s.validator_status == "fail"]
    if not failures:
        return "✅ Pass (không cần sửa)"
    return f"🔧 FailResult → output thay bằng fix_value ({failures[0].failure_reason})"


def _indent(text: str) -> str:
    """Thụt lề các dòng sau của output nhiều dòng để log dễ đọc."""
    return str(text).replace("\n", "\n          ")


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    """Chạy PIIDetector trên 7 test case: từng loại PII, nhiều PII, và text sạch."""
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    # on_fail truyền vào CONSTRUCTOR của validator, không phải Guard.use()
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    # Toàn bộ là dữ liệu PII giả (example.com, số 555, SSN/thẻ test)
    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Card + SSN",   "Card 4111-1111-1111-1111 is linked to SSN 987-65-4321."),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        print(f"\n[{label}]")
        result = guard.validate(text)
        print(f"  Status: {describe_outcome(result)}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    """Chạy JSONFormatter trên 6 test case: hợp lệ, từng loại lỗi sửa được, lỗi kết hợp, và không sửa được."""
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    # on_fail truyền vào CONSTRUCTOR của validator, không phải Guard.use()
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Combined errors",  "```json\n{'items': [1, 2, 3,], 'ok': true,}\n```"),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        print(f"\n[{label}]")
        result = guard.validate(text)
        print(f"  Status: {describe_outcome(result)}")
        print(f"  Input:  {_indent(text)}")
        print(f"  Output: {_indent(result.validated_output)}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main(demo: str = "all"):
    """
    Chạy demo Guardrails.

    demo: "pii" | "json" | "all" — tách riêng để lưu 2 file evidence:
        python 04_guardrails_validator.py --demo pii  > ../evidence/04_pii_demo_log.txt
        python 04_guardrails_validator.py --demo json > ../evidence/04_json_demo_log.txt
    """
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    if demo in ("pii", "all"):
        demo_pii_guard()
    if demo in ("json", "all"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bước 4: Guardrails AI Validators")
    parser.add_argument("--demo", choices=["pii", "json", "all"], default="all")
    main(parser.parse_args().demo)
