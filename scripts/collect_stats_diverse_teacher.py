"""Collect a bounded, auditable teacher corpus for the diverse statistics run.

The default mode is validation-only and never reads a credential.  ``--run``
is the only mode that calls OpenRouter.  Every paid response is written before
the next request, and an unresolved write-ahead marker prevents accidental
duplicate billing after an ambiguous interruption.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

from stats_consolidation_grader import grader_fingerprint, score, validate_raw_expression
from stats_consolidation_semantics import validate_question
from stats_teacher_envelope import _unique_object


MODEL = "meta-llama/llama-3.3-70b-instruct"
PACKET_SIZE = 12
EXPECTED_SPLIT_COUNTS = {"train": 864, "development": 108}
EXPECTED_CATEGORY_COUNTS = {"train": 48, "development": 6}
EXPECTED_CATEGORIES = (
    "both", "neither", "exactly_one", "at_least_one", "same",
    "moment_mean", "moment_variance", "moment_second",
    "poisson_variance", "poisson_scaled", "poisson_second",
    "process_variance", "process_scaled", "process_second",
    "uniform_mean", "uniform_conditional", "binomial", "interval",
)
MAX_CALLS = 144
MAX_COST_USD = 0.30
MAX_OUTPUT_TOKENS = 900
RESULT_NAME = "STATS_DIVERSE_TEACHER_RESULTS.json"

SYSTEM_PROMPT = (
    "Answer the supplied statistics questions independently. Return ONLY one JSON object with exactly "
    "this shape: {\"answers\":[{\"question_id\":\"...\",\"expression\":\"...\"}]}. "
    "Return exactly one item for every supplied question_id, in the supplied order. Each expression must be "
    "a single-line, fully numerical, executable expression. Keep arithmetic unevaluated. Use only numerical "
    "literals, parentheses, +, -, *, /, **, and comb(n,r). Do not use variables, prose, Markdown, code fences, "
    "E[...] or Var(...). Do not include any keys other than answers, question_id, and expression."
)

RULE_CARD = (
    "Generic rule card (apply only the relevant rule and substitute the question's own numbers): "
    "for independent events, both=p*q, neither=(1-p)*(1-q), exactly_one=p*(1-q)+(1-p)*q, "
    "at_least_one=1-(1-p)*(1-q), same=p*q+(1-p)*(1-q). For Y=a*X+b, "
    "E[Y]=a*E[X]+b, Var(Y)=a**2*Var(X), and E[Y**2]=Var(Y)+E[Y]**2. "
    "For a Poisson variable, variance equals mean; for a homogeneous Poisson process, mean and variance equal "
    "rate times duration after unit conversion. A uniform total conditional mean above c is (c+upper)/2; "
    "an unconditional uniform mean is (lower+upper)/2. Binomial P(X=r)=comb(n,r)*p**r*(1-p)**(n-r). "
    "For a symmetric interval, keep its center and divide its half-width by the stated square-root sample-size factor."
)


class CollectionStopped(RuntimeError):
    """A durable hard stop: no later request may be issued in this invocation."""


class SafePause(RuntimeError):
    """A test/integration pause raised only before creating an inflight marker."""


@contextmanager
def exclusive_run_lock(root: Path):
    """Serialize every reader/writer of a paid-call root, including after crashes.

    ``O_EXCL`` decides who creates the lock inode. ``flock`` is then the
    lifetime lock: the inode may remain after a crash, but the kernel releases
    the advisory lock, so a later invocation can recover without guessing
    whether the old PID is alive.
    """
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    path = root / "collection.lock"
    flags = os.O_RDWR | os.O_CREAT
    try:
        descriptor = os.open(path, flags | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another teacher-collection process holds this root lock") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, canonical_bytes({"pid": os.getpid(), "acquired_unix": time.time()}) + b"\n")
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_once(path: Path, value: Any) -> None:
    """Create immutable JSON with a race-free existence decision.

    A partial file left by an interrupted writer deliberately remains a hard
    stop: replacing it automatically could erase the only evidence that a
    paid request was already reserved or answered.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        old = read_json(path)
        if old != value:
            raise RuntimeError(f"Immutable evidence differs: {path}")
        return
    try:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_hash(record: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return digest(unsigned)


def _validate_record(record: dict[str, Any], path: Path) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"Malformed attempt evidence: {path}")
    if record.get("record_sha256") != _record_hash(record):
        raise RuntimeError(f"Attempt evidence hash mismatch: {path}")
    if record.get("request_sha256") != digest(record.get("request")):
        raise RuntimeError(f"Attempt request hash mismatch: {path}")


def _validate_request_evidence(record: dict[str, Any], path: Path) -> None:
    required = {"tag", "packet_index", "attempt", "requested_ids", "request", "request_sha256"}
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError(f"Malformed immutable request evidence: {path}")
    if record["request_sha256"] != digest(record["request"]):
        raise RuntimeError(f"Request evidence hash mismatch: {path}")
    if path.stem != record["tag"]:
        raise RuntimeError(f"Request evidence tag/filename mismatch: {path}")


def _evidence_tree(root: Path) -> dict[str, Any]:
    """Bind every immutable request and attempt file, including failed calls."""
    entries = []
    for folder, validator in (
        ("requests", _validate_request_evidence),
        ("attempts", _validate_record),
        ("pending_responses", _validate_pending_response),
    ):
        directory = root / folder
        for path in sorted(directory.glob("*.json")) if directory.exists() else []:
            validator(read_json(path), path)
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
    return {"files": entries, "file_count": len(entries), "tree_sha256": digest(entries)}


def load_curriculum(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError("Curriculum must be a JSON object")
    required = set(EXPECTED_SPLIT_COUNTS)
    if not required.issubset(document):
        raise ValueError("Curriculum must contain train and development")
    if set(document) != {"train", "development", "manifest"}:
        raise ValueError("Curriculum root must contain only train, development, and manifest")
    if not isinstance(document["manifest"], dict):
        raise ValueError("Curriculum manifest must be an object")
    manifest = document["manifest"]
    if manifest.get("counts", {}).get("train") != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError("Curriculum manifest train count mismatch")
    if manifest.get("counts", {}).get("development") != EXPECTED_SPLIT_COUNTS["development"]:
        raise ValueError("Curriculum manifest development count mismatch")
    verification_contract = manifest.get("verification_contract")
    if not isinstance(verification_contract, dict) or verification_contract.get("all_rows_verified") is not True:
        raise ValueError("Curriculum verification contract is missing or incomplete")
    if verification_contract.get("grader_sha256") != grader_fingerprint():
        raise ValueError("Curriculum grader commitment differs from the active grader")
    checker_path = Path(__file__).with_name("prepare_stats_diverse_curriculum.py")
    if verification_contract.get("builder_checker_source_sha256") != file_sha256(checker_path):
        raise ValueError("Curriculum checker commitment differs from the active checker")
    if manifest.get("categories") != list(EXPECTED_CATEGORIES):
        raise ValueError("Curriculum category contract differs from the fixed 18-category scope")
    isolation = manifest.get("isolation", {})
    if (isolation.get("teacher_allowed_input_splits") != ["train"]
            or isolation.get("development_used_for_training") is not False
            or isolation.get("final_blind_used_for_training") is not False):
        raise ValueError("Curriculum teacher/split isolation contract failed")
    blind_commitment = manifest.get("final_blind_commitment", {})
    if not isinstance(blind_commitment, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(blind_commitment.get("artifact_file_sha256", ""))
    ):
        raise ValueError("Sealed final-blind commitment is missing")
    splits: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for split in EXPECTED_SPLIT_COUNTS:
        rows = document[split]
        if not isinstance(rows, list) or len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected {split} size")
        categories = Counter()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row in {split}")
            missing = {
                "id", "category", "question", "prompt", "expression", "answer", "target", "semantics", "bindings"
            } - set(row)
            if missing:
                raise ValueError(f"Missing fields in {row.get('id', split)}: {sorted(missing)}")
            if not all(isinstance(row[key], str) and row[key]
                       for key in ("id", "category", "question", "prompt", "expression", "target")):
                raise ValueError(f"Invalid string field in {row.get('id', split)}")
            if row["id"] in all_ids:
                raise ValueError("Duplicate ID across curriculum: " + row["id"])
            all_ids.add(row["id"])
            categories[row["category"]] += 1
            if split == "train":
                binding = row.get("teacher_binding")
                if (not isinstance(binding, dict)
                        or binding.get("training_row_id") != row["id"]
                        or binding.get("result_id") != "teacher_result_" + row["id"]
                        or binding.get("request_prompt_sha256") != hashlib.sha256(row["prompt"].encode()).hexdigest()
                        or binding.get("canonical_target_sha256") != hashlib.sha256(row["target"].encode()).hexdigest()
                        or binding.get("required_release_state") != "accepted_or_oracle_corrected"
                        or binding.get("required_review_required") is not False
                        or row.get("teacher_eligible") is not True
                        or row.get("training_eligible") is not True
                        or row.get("teacher_exposed") is not False):
                    raise ValueError("Training row teacher binding is invalid: " + row["id"])
            validate_question(row)
            reference = score("Expression: " + row["expression"], row)
            if not (reference["math_correct"] is True and reference["executable"] is True
                    and reference.get("whole_raw_executable") is True
                    and reference["strict_one_line_expression"] is True):
                raise ValueError("Reference does not pass all teacher gates: " + row["id"])
        if set(categories) != set(EXPECTED_CATEGORIES) or set(categories.values()) != {EXPECTED_CATEGORY_COUNTS[split]}:
            raise ValueError(f"Unexpected per-category counts in {split}: {dict(categories)}")
        splits[split] = rows
    declared_split_hashes = manifest.get("split_sha256", {})
    if any(declared_split_hashes.get(split) != digest(splits[split]) for split in EXPECTED_SPLIT_COUNTS):
        raise ValueError("Curriculum split hash commitment mismatch")
    return document, splits


def sanitized_questions(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """The sole per-question teacher interface; deliberately drops every label."""
    return [{"id": row["id"], "question": row["question"]} for row in rows]


def make_request(rows: list[dict[str, Any]], retry: bool) -> dict[str, Any]:
    system = SYSTEM_PROMPT + ((" " + RULE_CARD) if retry else "")
    user = {"questions": sanitized_questions(rows)}
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "provider": {
            # The model remains pinned while OpenRouter may fail over between
            # providers inside this single paid request. The actual provider is
            # subsequently bound from the generation receipt.
            "allow_fallbacks": True,
            "require_parameters": True,
            "enforce_distillable_text": True,
            "max_price": {"prompt": 1, "completion": 2, "request": 0},
        },
    }


def request_cost_upper_bound(request: dict[str, Any]) -> float:
    """Conservatively reserve the provider-declared maximum before a call.

    UTF-8 bytes are an upper bound on prompt tokens for this ASCII-heavy
    protocol, so this intentionally over-reserves rather than risking a budget
    crossing. OpenRouter ``max_price`` values are US dollars per million tokens.
    """
    prices = request["provider"]["max_price"]
    prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in request["messages"])
    # Reserve additional chat-framing/special-token overhead that is not
    # represented in message content bytes.
    prompt_tokens_upper = prompt_bytes + 512
    return (
        prompt_tokens_upper * float(prices["prompt"])
        + int(request["max_tokens"]) * float(prices["completion"])
    ) / 1_000_000 + float(prices.get("request", 0))


def _forbidden_values(document: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> set[str]:
    values: set[str] = set()
    for split in ("development",):
        values.add(digest(splits[split]))
        for row in splits[split]:
            values.add(row["id"])
            values.add(row["question"])
            if isinstance(row.get("prompt"), str) and row["prompt"]:
                values.add(row["prompt"])
    def committed_hashes(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                committed_hashes(child)
        elif isinstance(value, list):
            for child in value:
                committed_hashes(child)
        elif isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            values.add(value)

    # Only inspect the explicit split commitments. Historical-registry metadata
    # is deliberately out of scope (and contains ordinary strings such as
    # "json" that are not secrets).
    manifest = document["manifest"]
    for registry_name in ("split_sha256", "order_sha256", "semantic_task_key_sha256",
                          "group_story_key_sha256"):
        registry = manifest.get(registry_name, {})
        if isinstance(registry, dict):
            committed_hashes(registry.get("development"))
    committed_hashes(manifest.get("final_blind_commitment", {}))
    return values


def audit_request(request: dict[str, Any], expected_rows: list[dict[str, Any]], forbidden: set[str]) -> None:
    if request.get("model") != MODEL or request.get("temperature") != 0:
        raise ValueError("Teacher identity or temperature changed")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Unexpected teacher message count")
    body = json.loads(messages[1]["content"], object_pairs_hook=_unique_object)
    if set(body) != {"questions"} or body["questions"] != sanitized_questions(expected_rows):
        raise ValueError("Teacher payload contains fields beyond train id/question")
    payload_text = canonical_bytes(request).decode("utf-8")
    collision = next((value for value in forbidden if value and value in payload_text), None)
    if collision is not None:
        raise ValueError("Development/final content leaked into teacher request")
    # Specific label names are forbidden in the user payload even if a future
    # curriculum happens to add one under an unexpected spelling.
    forbidden_keys = {"oracle", "reference", "expression", "answer", "target", "development", "final", "prompt"}
    for item in body["questions"]:
        if set(item) != {"id", "question"} or forbidden_keys.intersection(item):
            raise ValueError("Teacher question payload is not sanitized")


def parse_response(raw: Any, expected_ids: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    parsed = {"ok": False, "strict_envelope": False, "error": None, "returned_ids": []}
    if not isinstance(raw, str):
        parsed["error"] = "non_text_response"
        return {}, parsed
    try:
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
        obj, end = decoder.raw_decode(raw.strip())
        if raw.strip()[end:].strip():
            raise ValueError("Trailing content outside JSON")
        if not isinstance(obj, dict) or set(obj) != {"answers"} or not isinstance(obj["answers"], list):
            raise ValueError("Envelope must contain only an answers list")
        if not expected_ids or len(expected_ids) != len(set(expected_ids)):
            raise ValueError("Expected question IDs must be unique and nonempty")
        allowed = set(expected_ids)
        answers: dict[str, str] = {}
        returned_order: list[str] = []
        for item in obj["answers"]:
            if not isinstance(item, dict) or set(item) != {"question_id", "expression"}:
                raise ValueError("Answer item has unexpected fields")
            qid, expression = item["question_id"], item["expression"]
            if not isinstance(qid, str) or qid not in allowed or qid in answers:
                raise ValueError("Duplicate or unexpected question_id")
            if (not isinstance(expression, str) or not expression or expression != expression.strip()
                    or "\n" in expression or "\r" in expression):
                raise ValueError("Expression must be a nonempty trimmed single line")
            answers[qid] = expression
            returned_order.append(qid)
        if returned_order != expected_ids:
            raise ValueError("Answer IDs must exactly match the supplied IDs in supplied order")
        parsed.update(ok=True, strict_envelope=True, returned_ids=returned_order)
        return answers, parsed
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        parsed["error"] = str(exc)
        return {}, parsed


def judge_response(raw: Any, rows: list[dict[str, Any]], finish_reason: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = [row["id"] for row in rows]
    expressions, parsed = parse_response(raw, ids)
    decisions = []
    for row in rows:
        expression = expressions.get(row["id"])
        target = "Expression: " + expression if expression is not None else None
        if finish_reason != "stop":
            judged = {"math_correct": None, "executable": False, "strict_one_line_expression": False,
                      "review_required": True, "reason": "finish_reason_not_stop"}
        elif not parsed["ok"] or expression is None:
            judged = {"math_correct": None, "executable": False, "strict_one_line_expression": False,
                      "review_required": True, "reason": parsed["error"] or "missing_question_id"}
        else:
            try:
                unchanged_expression, exact_value = validate_raw_expression(expression)
                if unchanged_expression != expression:
                    raise ValueError("Teacher expression field must not contain an Expression: prefix")
            except (ValueError, SyntaxError, OverflowError, RecursionError) as exc:
                judged = {
                    "math_correct": None,
                    "executable": False,
                    "strict_one_line_expression": False,
                    "review_required": True,
                    "reason": "raw_expression_contract_failed: " + str(exc),
                }
            else:
                judged = score(target, row)
                judged["raw_expression_contract_verified"] = True
                judged["raw_expression_exact_value"] = exact_value
        gates = {
            "math_correct": judged.get("math_correct") is True,
            "executable": judged.get("executable") is True,
            "whole_raw_executable": bool(
                judged.get("whole_raw_executable") is True
                and judged.get("raw_expression_contract_verified") is True
            ),
            "strict_format": bool(parsed["strict_envelope"] and judged.get("strict_one_line_expression") is True),
        }
        # ``math_correct`` comes only from the frozen structural-equivalence
        # bank plus exact symbolic agreement; numerical coincidence alone is
        # pending/false and cannot pass this gate.
        canonical_exact = target == row["target"]
        accepted = all(gates.values())
        if accepted:
            classification = "accepted"
        elif judged.get("math_correct") is None and judged.get("executable"):
            classification = "equivalence_pending"
        elif judged.get("math_correct") is False:
            classification = "mathematical_error"
        else:
            classification = "parse_or_format_error"
        decisions.append({
            "id": row["id"],
            "expression": expression,
            "training_target": target,
            "gates": gates,
            "canonical_target_exact": canonical_exact,
            "accepted": accepted,
            "classification": classification,
            "grader": judged,
        })
    return parsed, decisions


def openrouter_transport(request: dict[str, Any], api_key: str) -> dict[str, Any]:
    import requests

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + api_key},
            json=request,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Teacher network failure; inflight marker retained") from exc
    if response.status_code != 200:
        raise RuntimeError(f"Teacher HTTP {response.status_code}; inflight marker retained; response body omitted")
    response_text = getattr(response, "text", None)
    try:
        body = response.json()
    except (ValueError, TypeError):
        # A paid HTTP-200 response must still be durably preserved and must not
        # be silently retried merely because its JSON envelope was malformed.
        body = None
    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
    if not isinstance(response_text, str):
        response_text = json.dumps(body, ensure_ascii=False, sort_keys=True) if body is not None else ""
    return {
        "raw": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "model": body.get("model") if isinstance(body, dict) else None,
        # Some live responses include this legacy convenience field, but the
        # documented ChatResult schema does not guarantee it. Missing provider
        # is resolved through the Generation endpoint after persistence.
        "provider": body.get("provider") if isinstance(body, dict) else None,
        "usage": usage,
        "response_id": body.get("id") if isinstance(body, dict) else None,
        "api_response_body": body,
        "api_response_text": response_text,
        "api_response_text_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
    }


def openrouter_generation_transport(response_id: str, api_key: str) -> dict[str, Any]:
    """Fetch the documented provider/cost receipt for one completed generation."""
    import requests

    last_status = None
    for attempt in range(5):
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/generation",
                headers={"Authorization": "Bearer " + api_key},
                params={"id": response_id},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Generation-metadata network failure") from exc
        last_status = response.status_code
        if response.status_code == 200:
            try:
                body = response.json()
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Generation-metadata response is not JSON") from exc
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, dict):
                raise RuntimeError("Generation-metadata response lacks data")
            return {"response_body": body, "data": data}
        # Generation statistics may be briefly unavailable just after the chat
        # response. These GET retries cannot create another paid generation.
        if response.status_code in {404, 425, 429} and attempt < 4:
            time.sleep(1.0)
            continue
        break
    raise RuntimeError(f"Generation-metadata HTTP {last_status}")


def _attempt_paths(root: Path, tag: str) -> tuple[Path, Path]:
    return root / "requests" / f"{tag}.json", root / "attempts" / f"{tag}.json"


def _pending_response_path(root: Path, tag: str) -> Path:
    return root / "pending_responses" / f"{tag}.json"


def _pending_response_hash(record: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "pending_response_sha256"}
    return digest(unsigned)


def _validate_pending_response(record: dict[str, Any], path: Path) -> None:
    required = {"tag", "request_sha256", "transport_response", "transport_response_sha256",
                "pending_response_sha256"}
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError(f"Malformed pending paid response: {path}")
    if record.get("tag") != path.stem:
        raise RuntimeError(f"Pending paid response tag/filename mismatch: {path}")
    if record.get("transport_response_sha256") != digest(record.get("transport_response")):
        raise RuntimeError(f"Pending paid response transport hash mismatch: {path}")
    if record.get("pending_response_sha256") != _pending_response_hash(record):
        raise RuntimeError(f"Pending paid response record hash mismatch: {path}")


def _all_attempts(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "attempts").glob("*.json")) if (root / "attempts").exists() else []:
        record = read_json(path)
        _validate_record(record, path)
        request_path, _ = _attempt_paths(root, record["tag"])
        if not request_path.exists():
            raise RuntimeError(f"Attempt has no immutable request evidence: {path}")
        request_evidence = read_json(request_path)
        _validate_request_evidence(request_evidence, request_path)
        expected = {key: record[key] for key in
                    ("tag", "packet_index", "attempt", "requested_ids", "request", "request_sha256")}
        if request_evidence != expected:
            raise RuntimeError(f"Request/attempt evidence mismatch: {path}")
        records.append(record)
    return records


def _all_pending_responses(root: Path) -> list[dict[str, Any]]:
    records = []
    directory = root / "pending_responses"
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        record = read_json(path)
        _validate_pending_response(record, path)
        records.append(record)
    return records


def _usage_state(root: Path) -> dict[str, Any]:
    attempts = _all_attempts(root)
    pending = _all_pending_responses(root)
    costs = [record.get("cost_usd") for record in attempts]
    for record in pending:
        response = record.get("transport_response")
        usage = response.get("usage") if isinstance(response, dict) else None
        costs.append(usage.get("cost") if isinstance(usage, dict) else None)
    return {
        "calls": len(attempts) + len(pending),
        "finalized_calls": len(attempts),
        "pending_paid_responses": len(pending),
        "reported_cost_usd": sum(cost for cost in costs if type(cost) in (int, float)),
        "missing_cost": any(type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0 for cost in costs),
    }


def _enforce_completed_attempt(root: Path, record: dict[str, Any]) -> None:
    """Apply identical durable hard stops to new and cached responses."""
    cost = record.get("cost_usd")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise CollectionStopped("Completed response lacks valid client-reported cost")
    now = _usage_state(root)
    if now["reported_cost_usd"] > MAX_COST_USD:
        raise CollectionStopped("Client-reported cost exceeded the US$0.30 hard cap")
    if record.get("model") != MODEL:
        raise CollectionStopped("Provider returned a model other than the pinned teacher")
    if not isinstance(record.get("provider"), str) or not record["provider"]:
        raise CollectionStopped("Provider identity is missing")
    if not isinstance(record.get("response_id"), str) or not record["response_id"]:
        raise CollectionStopped("Provider response ID is missing")
    provider_receipt = record.get("provider_receipt")
    if not isinstance(provider_receipt, dict) or provider_receipt.get("source") not in {
        "chat_completion_legacy_field", "openrouter_generation_endpoint",
    }:
        raise CollectionStopped("Provider provenance receipt is missing")
    if record.get("transport_response_sha256") != digest(record.get("transport_response")):
        raise CollectionStopped("Persisted paid transport response hash mismatch")


def _reconcile_inflight(root: Path) -> None:
    marker_path = root / "inflight.json"
    marker = read_json(marker_path)
    if marker is None:
        pending = _all_pending_responses(root)
        if pending:
            raise RuntimeError("Pending paid response has no matching inflight marker")
        return
    if (not isinstance(marker, dict) or set(marker) != {"tag", "request_sha256", "created_unix"}
            or not isinstance(marker.get("tag"), str)
            or not isinstance(marker.get("request_sha256"), str)):
        raise RuntimeError("Malformed inflight marker")
    _, attempt_path = _attempt_paths(root, marker["tag"])
    pending_path = _pending_response_path(root, marker["tag"])
    if attempt_path.exists():
        record = read_json(attempt_path)
        _validate_record(record, attempt_path)
        if record["request_sha256"] != marker["request_sha256"]:
            raise RuntimeError("Inflight marker does not match durable response")
        if pending_path.exists():
            pending = read_json(pending_path)
            _validate_pending_response(pending, pending_path)
            if (pending["request_sha256"] != marker["request_sha256"]
                    or record.get("transport_response_sha256") != pending["transport_response_sha256"]):
                raise RuntimeError("Final attempt does not match pending paid response")
            pending_path.unlink()
        marker_path.unlink()
        return
    if pending_path.exists():
        pending = read_json(pending_path)
        _validate_pending_response(pending, pending_path)
        if pending["request_sha256"] != marker["request_sha256"]:
            raise RuntimeError("Inflight marker does not match pending paid response")
        # Recoverable: _execute_attempt will finalize this already-paid response
        # and must not call the chat-completions endpoint again.
        return
    raise RuntimeError("Unresolved inflight request; refusing possible duplicate billing")


def _execute_attempt(
    root: Path,
    tag: str,
    packet_index: int,
    attempt_index: int,
    rows: list[dict[str, Any]],
    retry: bool,
    api_key: str,
    transport: Callable[[dict[str, Any], str], dict[str, Any]],
    provider_lookup: Callable[[str, str], dict[str, Any]],
    forbidden: set[str],
) -> dict[str, Any]:
    request_path, attempt_path = _attempt_paths(root, tag)
    request = make_request(rows, retry)
    audit_request(request, rows, forbidden)
    request_evidence = {
        "tag": tag,
        "packet_index": packet_index,
        "attempt": attempt_index,
        "requested_ids": [row["id"] for row in rows],
        "request": request,
        "request_sha256": digest(request),
    }
    write_once(request_path, request_evidence)
    if attempt_path.exists():
        old = read_json(attempt_path)
        _validate_record(old, attempt_path)
        if old["request_sha256"] != request_evidence["request_sha256"]:
            raise RuntimeError("Cached response belongs to a different request")
        _enforce_completed_attempt(root, old)
        return old

    _reconcile_inflight(root)
    marker_path = root / "inflight.json"
    pending_path = _pending_response_path(root, tag)
    marker = read_json(marker_path)
    pending = read_json(pending_path)
    if marker is not None:
        if marker.get("tag") != tag or marker.get("request_sha256") != request_evidence["request_sha256"]:
            raise RuntimeError("A different paid request remains inflight")
        if pending is None:
            raise RuntimeError("Unresolved inflight request; refusing possible duplicate billing")
        _validate_pending_response(pending, pending_path)
        response = pending["transport_response"]
    else:
        if pending is not None:
            raise RuntimeError("Pending paid response has no matching inflight marker")
        usage = _usage_state(root)
        if usage["missing_cost"]:
            raise CollectionStopped("A prior completed response lacks valid client-reported cost")
        if usage["calls"] >= MAX_CALLS or usage["reported_cost_usd"] >= MAX_COST_USD:
            raise CollectionStopped("Persistent teacher call/cost cap reached")
        reservation = request_cost_upper_bound(request)
        if usage["reported_cost_usd"] + reservation > MAX_COST_USD:
            raise CollectionStopped(
                "Next request's conservative maximum-cost reservation would exceed the US$0.30 hard cap"
            )

        marker = {
            "tag": tag,
            "request_sha256": request_evidence["request_sha256"],
            "created_unix": time.time(),
        }
        # This must be an exclusive create, not an exists-then-replace write:
        # observing another marker is a stop condition before any paid call.
        write_once(marker_path, marker)
        append_jsonl(root / "journal.jsonl", {"event": "started", **marker})
        response = transport(request, api_key)
        if not isinstance(response, dict):
            raise CollectionStopped("Paid response transport returned a non-object; inflight marker retained")
        pending = {
            "tag": tag,
            "request_sha256": request_evidence["request_sha256"],
            "transport_response": response,
            "transport_response_sha256": digest(response),
        }
        pending["pending_response_sha256"] = _pending_response_hash(pending)
        write_once(pending_path, pending)
        append_jsonl(root / "journal.jsonl", {
            "event": "paid_response_persisted_before_metadata_lookup",
            "tag": tag,
            "request_sha256": request_evidence["request_sha256"],
            "pending_response_sha256": pending["pending_response_sha256"],
        })

    raw = response.get("raw")
    usage_payload = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    chat_cost = usage_payload.get("cost")
    provider = response.get("provider")
    provider_receipt: dict[str, Any] = {
        "source": "chat_completion_legacy_field",
        "generation_response": None,
    }
    cost_candidates = [chat_cost] if (
        type(chat_cost) in (int, float) and math.isfinite(chat_cost) and chat_cost >= 0
    ) else []
    if not isinstance(provider, str) or not provider or not cost_candidates:
        response_id = response.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise CollectionStopped(
                "Paid response saved but lacks a response ID needed for provider/cost reconciliation; "
                "inflight marker retained"
            )
        try:
            generation = provider_lookup(response_id, api_key)
        except Exception as exc:
            raise CollectionStopped(
                "Paid response saved; generation metadata lookup failed; resume will reconcile without resending"
            ) from exc
        data = generation.get("data") if isinstance(generation, dict) else None
        if not isinstance(data, dict):
            raise CollectionStopped(
                "Paid response saved; generation metadata is malformed; inflight marker retained"
            )
        if data.get("id") != response_id:
            raise CollectionStopped(
                "Paid response saved; generation metadata ID mismatch; inflight marker retained"
            )
        generation_provider = data.get("provider_name")
        generation_cost = data.get("total_cost")
        if not isinstance(generation_provider, str) or not generation_provider:
            raise CollectionStopped(
                "Paid response saved; generation metadata lacks provider identity; inflight marker retained"
            )
        if (type(generation_cost) not in (int, float) or not math.isfinite(generation_cost)
                or generation_cost < 0):
            raise CollectionStopped(
                "Paid response saved; generation metadata lacks valid total_cost; inflight marker retained"
            )
        if isinstance(data.get("model"), str) and data["model"] != response.get("model"):
            raise CollectionStopped(
                "Paid response saved; chat/generation model receipts disagree; inflight marker retained"
            )
        provider = generation_provider
        cost_candidates.append(generation_cost)
        provider_receipt = {
            "source": "openrouter_generation_endpoint",
            "generation_response": generation,
        }
    cost = max(cost_candidates) if cost_candidates else None
    parsed, decisions = judge_response(raw, rows, response.get("finish_reason"))
    record = {
        **request_evidence,
        "transport_response": response,
        "transport_response_sha256": digest(response),
        "raw": raw,
        "model": response.get("model"),
        "provider": provider,
        "provider_receipt": provider_receipt,
        "usage": usage_payload,
        "cost_usd": cost,
        "finish_reason": response.get("finish_reason"),
        "response_id": response.get("response_id"),
        "parse": parsed,
        "decisions": decisions,
    }
    record["record_sha256"] = _record_hash(record)
    write_once(attempt_path, record)
    append_jsonl(root / "journal.jsonl", {
        "event": "response_persisted",
        "tag": tag,
        "request_sha256": record["request_sha256"],
        "record_sha256": record["record_sha256"],
        "cost_usd": cost,
    })
    pending_path.unlink()
    marker_path.unlink()

    # Identity/cost failures are persisted with parse and all three gates, but
    # become a hard stop before another call can be made.
    _enforce_completed_attempt(root, record)
    return record


def _manifest(curriculum_path: Path, document: dict[str, Any], train: list[dict[str, Any]]) -> dict[str, Any]:
    planned = [make_request(train[start:start + PACKET_SIZE], False)
               for start in range(0, len(train), PACKET_SIZE)]
    return {
        "version": "stats-diverse-teacher-v1",
        # A logical name is portable between Kaggle and the local audit host;
        # the whole-file hash below, not an environment-specific absolute path,
        # is the identity lock.
        "curriculum_path": curriculum_path.name,
        "curriculum_sha256": file_sha256(curriculum_path),
        "curriculum_canonical_sha256": digest(document),
        "train_sha256": digest(train),
        "request_sha256": digest(planned),
        "request_contract_sha256": digest({
            "model": MODEL,
            "temperature": 0,
            "packet_size": PACKET_SIZE,
            "system_prompt": SYSTEM_PROMPT,
            "retry_rule_card": RULE_CARD,
            "max_calls": MAX_CALLS,
            "max_cost_usd": MAX_COST_USD,
        }),
        "collector_sha256": file_sha256(Path(__file__)),
        "grader_sha256": grader_fingerprint(),
        "teacher_model": MODEL,
        "temperature": 0,
        "packet_size": PACKET_SIZE,
        "initial_packet_count": len(planned),
    }


def _canonical_receipt(question: dict[str, Any]) -> dict[str, bool]:
    """Re-run the frozen prompt, oracle, grader, and mutation checks."""
    from prepare_stats_diverse_curriculum import canonical_semantics, parse_question_semantics, verify_row

    rebuilt = copy.deepcopy(question)
    verify_row(rebuilt)
    parsed_semantics = parse_question_semantics(
        question["question"],
        question["category"],
        question["coverage_axes"]["surface_form"],
    )
    independently_parsed_prompt = canonical_semantics(parsed_semantics) == canonical_semantics(question["semantics"])
    judged = score("Expression: " + question["expression"], question)
    noncolliding = [
        trap for trap in rebuilt.get("misconception_traps", [])
        if trap.get("collides_with_oracle") is False
    ]
    mutations_rejected = bool(noncolliding) and all(
        not score("Expression: " + trap["expression"], question).get("primary_correct", False)
        for trap in noncolliding
    )
    frozen_receipts_match = (
        rebuilt.get("verification") == question.get("verification")
        and rebuilt.get("misconception_traps") == question.get("misconception_traps")
    )
    return {
        "prompt_semantics_verified": bool(
            frozen_receipts_match
            and independently_parsed_prompt
            and rebuilt.get("verification", {}).get("prompt_semantics_consistent") is True
        ),
        "symbolic_oracle_verified": bool(
            frozen_receipts_match
            and judged.get("math_correct") is True
            and judged.get("executable") is True
            and judged.get("whole_raw_executable") is True
            and judged.get("strict_one_line_expression") is True
            and rebuilt.get("verification", {}).get("independent_oracle_correct") is True
            and rebuilt.get("verification", {}).get("calculator_matches_oracle") is True
            and rebuilt.get("verification", {}).get("grader_primary_correct") is True
        ),
        "mutation_verified": bool(frozen_receipts_match and mutations_rejected),
    }


def _validate_attempt_derivations(
    attempts: list[dict[str, Any]],
    train: list[dict[str, Any]],
    forbidden: set[str],
) -> None:
    """Recompute every derived decision from immutable request/response bytes."""
    by_id = {row["id"]: row for row in train}
    by_tag = {record.get("tag"): record for record in attempts}
    if len(by_tag) != len(attempts):
        raise RuntimeError("Duplicate attempt tags")
    response_ids: set[str] = set()
    for record in attempts:
        packet_index = record.get("packet_index")
        attempt_index = record.get("attempt")
        if (type(packet_index) is not int or not 0 <= packet_index < len(train) // PACKET_SIZE
                or attempt_index not in (0, 1)):
            raise RuntimeError("Attempt packet/attempt index is outside the frozen plan")
        expected_tag = f"packet_{packet_index:03d}_attempt_{attempt_index}"
        if record.get("tag") != expected_tag:
            raise RuntimeError("Attempt tag does not match packet/attempt index")
        requested_ids = record.get("requested_ids")
        if (not isinstance(requested_ids, list) or not requested_ids
                or not all(isinstance(item, str) for item in requested_ids)
                or len(requested_ids) != len(set(requested_ids))
                or any(item not in by_id for item in requested_ids)):
            raise RuntimeError("Attempt requested_ids are malformed")
        packet = train[packet_index * PACKET_SIZE:(packet_index + 1) * PACKET_SIZE]
        packet_ids = [row["id"] for row in packet]
        if attempt_index == 0 and requested_ids != packet_ids:
            raise RuntimeError("First attempt does not bind the complete frozen packet")
        if attempt_index == 1 and any(item not in packet_ids for item in requested_ids):
            raise RuntimeError("Retry escaped its frozen packet")
        requested_rows = [by_id[item] for item in requested_ids]
        expected_request = make_request(requested_rows, attempt_index == 1)
        if record.get("request") != expected_request:
            raise RuntimeError("Attempt request differs from the frozen train-only request")
        audit_request(record["request"], requested_rows, forbidden)

        transport_response = record.get("transport_response")
        if not isinstance(transport_response, dict):
            raise RuntimeError("Attempt lacks the preserved paid transport response")
        if record.get("transport_response_sha256") != digest(transport_response):
            raise RuntimeError("Attempt transport-response hash mismatch")
        usage = transport_response.get("usage") if isinstance(transport_response.get("usage"), dict) else {}
        shared_response_fields = {
            "raw": transport_response.get("raw"),
            "model": transport_response.get("model"),
            "usage": usage,
            "finish_reason": transport_response.get("finish_reason"),
            "response_id": transport_response.get("response_id"),
        }
        if any(record.get(key) != value for key, value in shared_response_fields.items()):
            raise RuntimeError("Attempt fields differ from the preserved paid transport response")
        response_id = record.get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise RuntimeError("Attempt response IDs must be unique nonempty strings")
        response_ids.add(response_id)

        provider_receipt = record.get("provider_receipt")
        if not isinstance(provider_receipt, dict):
            raise RuntimeError("Attempt provider receipt is malformed")
        source = provider_receipt.get("source")
        chat_cost = usage.get("cost")
        costs = [chat_cost] if (
            type(chat_cost) in (int, float) and math.isfinite(chat_cost) and chat_cost >= 0
        ) else []
        if source == "chat_completion_legacy_field":
            if (provider_receipt.get("generation_response") is not None
                    or record.get("provider") != transport_response.get("provider")):
                raise RuntimeError("Legacy provider receipt differs from chat response")
        elif source == "openrouter_generation_endpoint":
            generation = provider_receipt.get("generation_response")
            data = generation.get("data") if isinstance(generation, dict) else None
            if (not isinstance(data, dict) or data.get("id") != response_id
                    or data.get("provider_name") != record.get("provider")):
                raise RuntimeError("Generation endpoint receipt does not bind provider/response ID")
            generation_cost = data.get("total_cost")
            if (type(generation_cost) not in (int, float) or not math.isfinite(generation_cost)
                    or generation_cost < 0):
                raise RuntimeError("Generation endpoint receipt has invalid total_cost")
            costs.append(generation_cost)
        else:
            raise RuntimeError("Unknown provider receipt source")
        if not costs or record.get("cost_usd") != max(costs):
            raise RuntimeError("Attempt cost does not match its provider receipts")

        parsed, decisions = judge_response(record.get("raw"), requested_rows, record.get("finish_reason"))
        if record.get("parse") != parsed or record.get("decisions") != decisions:
            raise RuntimeError("Attempt parse/decisions do not recompute from immutable raw response")

    for record in attempts:
        if record["attempt"] != 1:
            continue
        first_tag = f"packet_{record['packet_index']:03d}_attempt_0"
        first = by_tag.get(first_tag)
        if first is None:
            raise RuntimeError("Retry has no corresponding first attempt")
        failed_ids = [item["id"] for item in first["decisions"] if not item["accepted"]]
        if record["requested_ids"] != failed_ids:
            raise RuntimeError("Retry IDs are not exactly the first-attempt failures in packet order")


def _summarize(root: Path, manifest: dict[str, Any], train: list[dict[str, Any]],
               forbidden: set[str], stop_reason: str | None) -> dict[str, Any]:
    attempts = _all_attempts(root)
    _validate_attempt_derivations(attempts, train, forbidden)
    teacher_leakage_detected = any(
        value and (value in canonical_bytes(record["request"]).decode("utf-8")
                   or (isinstance(record.get("raw"), str) and value in record["raw"]))
        for record in attempts for value in forbidden
    )
    by_id: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in train}
    for record in attempts:
        for decision in record["decisions"]:
            if decision["id"] in by_id:
                by_id[decision["id"]].append({
                    "tag": record["tag"],
                    "attempt": record["attempt"],
                    **decision,
                })
    rows = []
    for question in train:
        history = sorted(by_id[question["id"]], key=lambda item: item["attempt"])
        accepted = next((item for item in history if item["accepted"]), None)
        selected = accepted or (history[-1] if history else None)
        first_finished = any(item["attempt"] == 0 for item in history)
        retry_finished = any(item["attempt"] == 1 for item in history)
        canonical = _canonical_receipt(question)
        corrected = accepted is None and first_finished and retry_finished and all(canonical.values())
        if accepted:
            status = "raw_teacher_accepted"
            expression = accepted["expression"]
            target = accepted["training_target"]
            provenance = {
                "kind": "raw_teacher_accepted",
                "attempt_tag": accepted["tag"],
                "raw_expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                "canonical_correction": False,
                "canonical_expression_exact": expression == question["expression"],
                "frozen_equivalence_bank_accepted": True,
            }
        elif corrected:
            status = "canonical_corrected"
            expression = question["expression"]
            target = "Expression: " + expression
            provenance = {
                "kind": "canonical_corrected",
                "attempt_tags": [item["tag"] for item in history],
                "canonical_expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                "canonical_correction": True,
                "teacher_final_classification": selected["classification"],
            }
        else:
            status = selected["classification"] if selected else "not_attempted"
            expression = None
            target = None
            provenance = None
        binding = question.get("teacher_binding") if isinstance(question.get("teacher_binding"), dict) else {}
        result_id = binding.get("result_id", "teacher_result_" + question["id"])
        raw_expression = selected.get("expression") if selected and isinstance(selected.get("expression"), str) else ""
        if target is not None:
            training_judgement = score(target, question)
            training_target_gates = {
                "math_correct": training_judgement.get("math_correct") is True,
                "executable": training_judgement.get("executable") is True,
                "whole_raw_executable": training_judgement.get("whole_raw_executable") is True,
                "strict_format": training_judgement.get("strict_one_line_expression") is True,
                "review_resolved": training_judgement.get("review_required") is False,
            }
        else:
            training_target_gates = {
                "math_correct": False,
                "executable": False,
                "whole_raw_executable": False,
                "strict_format": False,
                "review_resolved": False,
            }
        row_result = {
            "id": result_id,
            "result_id": result_id,
            "training_row_id": question["id"],
            "category": question["category"],
            "status": status,
            "raw_expression": raw_expression,
            "expression": expression,
            "training_target": target,
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest() if target else None,
            "canonical_reference_target": question["target"],
            "canonical_reference_target_sha256": hashlib.sha256(question["target"].encode("utf-8")).hexdigest(),
            "review_required": status not in {"raw_teacher_accepted", "canonical_corrected"},
            "target_provenance": provenance,
            "canonical_verification": canonical,
            "training_target_gates": training_target_gates,
            "attempt_count": len(history),
            "attempt_tags": [item["tag"] for item in history],
            "gates": selected["gates"] if selected else {
                "math_correct": False, "executable": False,
                "whole_raw_executable": False, "strict_format": False,
            },
        }
        row_result["result_sha256"] = digest(row_result)
        rows.append(row_result)
    counts = Counter(row["status"] for row in rows)
    usable_statuses = {"raw_teacher_accepted", "canonical_corrected"}
    usable_by_category = Counter(row["category"] for row in rows if row["status"] in usable_statuses)
    usage = _usage_state(root)
    pending = sum(row["status"] == "equivalence_pending" for row in rows)
    targets_trace_provenance = True
    for row in rows:
        if row["status"] not in usable_statuses:
            continue
        if row["training_target"] != "Expression: " + row["expression"] or row["target_provenance"] is None:
            targets_trace_provenance = False
        if not all(row["canonical_verification"].values()):
            targets_trace_provenance = False
        if not all(row["training_target_gates"].values()):
            targets_trace_provenance = False
    usable = counts["raw_teacher_accepted"] + counts["canonical_corrected"]
    by_category_status: dict[str, dict[str, int]] = {}
    for category in EXPECTED_CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        category_counts = Counter(row["status"] for row in category_rows)
        raw_count = category_counts["raw_teacher_accepted"]
        corrected_count = category_counts["canonical_corrected"]
        category_pending = category_counts["equivalence_pending"]
        by_category_status[category] = {
            "raw_teacher_accepted": raw_count,
            "canonical_corrected": corrected_count,
            "pending": category_pending,
            "failed_or_unattempted": len(category_rows) - raw_count - corrected_count - category_pending,
        }
    release = (
        usable == EXPECTED_SPLIT_COUNTS["train"]
        and pending == 0
        and usage["calls"] <= MAX_CALLS
        and not usage["missing_cost"]
        and usage["reported_cost_usd"] <= MAX_COST_USD
        and targets_trace_provenance
        and all(value == EXPECTED_CATEGORY_COUNTS["train"] for value in usable_by_category.values())
        and len(usable_by_category) == 18
        and all(record["model"] == MODEL and isinstance(record["provider"], str) and record["provider"]
                for record in attempts)
        and not teacher_leakage_detected
        and stop_reason is None
    )
    raw_total = counts["raw_teacher_accepted"]
    corrected_total = counts["canonical_corrected"]
    if release and raw_total == EXPECTED_SPLIT_COUNTS["train"] and corrected_total == 0:
        supervision_description = "pure_teacher_response_distillation"
    elif raw_total and corrected_total:
        supervision_description = "hybrid_teacher_response_and_canonical_oracle_supervision"
    elif corrected_total:
        supervision_description = "canonical_oracle_supervision_after_teacher_audit"
    elif raw_total:
        supervision_description = "partial_teacher_response_supervision_not_released"
    else:
        supervision_description = "no_released_supervision"
    evidence_tree = _evidence_tree(root)
    return {
        "version": "stats-diverse-teacher-results-v1",
        "training_release": release,
        "stop_reason": stop_reason,
        "receipts": {
            **manifest,
            "attempts_tree_sha256": evidence_tree["tree_sha256"],
            "teacher_evidence_tree": evidence_tree,
        },
        "counts": {
            "usable_targets": usable,
            "raw_teacher_accepted": counts["raw_teacher_accepted"],
            "canonical_corrected": counts["canonical_corrected"],
            "pending": pending,
            "failed_or_unattempted": len(train) - usable - pending,
            "by_status": dict(counts),
            "usable_by_category": dict(usable_by_category),
            "by_category_status": by_category_status,
        },
        "supervision_description": supervision_description,
        "supervision_proportions": {
            "denominator_train_rows": EXPECTED_SPLIT_COUNTS["train"],
            "raw_teacher_accepted": {
                "count": raw_total,
                "proportion": raw_total / EXPECTED_SPLIT_COUNTS["train"],
            },
            "canonical_corrected": {
                "count": corrected_total,
                "proportion": corrected_total / EXPECTED_SPLIT_COUNTS["train"],
            },
        },
        "usage": usage,
        "all_training_targets_have_explicit_provenance": targets_trace_provenance,
        "development_or_final_leakage_detected": teacher_leakage_detected,
        "rows": rows,
    }


def validate_existing(root: Path, manifest: dict[str, Any], train: list[dict[str, Any]],
                      forbidden: set[str]) -> dict[str, Any] | None:
    if not root.exists():
        return None
    _reconcile_inflight(root)
    locked = read_json(root / "manifest.json")
    if locked is not None and locked != manifest:
        raise RuntimeError("Run manifest/hash mismatch; refusing changed curriculum, request, or grader")
    attempts = _all_attempts(root)
    _validate_attempt_derivations(attempts, train, forbidden)
    result = read_json(root / RESULT_NAME)
    if result is not None:
        sidecar = root / (RESULT_NAME + ".sha256")
        if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != file_sha256(root / RESULT_NAME):
            raise RuntimeError("Teacher-results whole-file SHA-256 sidecar mismatch")
        tree = _evidence_tree(root)
        expected_receipts = {
            **manifest,
            "attempts_tree_sha256": tree["tree_sha256"],
            "teacher_evidence_tree": tree,
        }
        if result.get("receipts") != expected_receipts:
            raise RuntimeError("Result receipt mismatch")
        recomputed = _summarize(root, manifest, train, forbidden, result.get("stop_reason"))
        if result != recomputed:
            raise RuntimeError("Result does not match immutable attempt evidence")
    else:
        # Hash validity alone is insufficient: all parse/decision derivatives
        # above were freshly reconstructed from the preserved response bytes.
        pass
    return result


def run_collection(
    curriculum_path: Path,
    root: Path,
    api_key: str,
    transport: Callable[[dict[str, Any], str], dict[str, Any]] = openrouter_transport,
    provider_lookup: Callable[[str, str], dict[str, Any]] = openrouter_generation_transport,
    max_new_calls: int | None = None,
    expected_curriculum_sha256: str | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(root):
        return _run_collection_locked(
            curriculum_path,
            root,
            api_key,
            transport,
            provider_lookup,
            max_new_calls,
            expected_curriculum_sha256,
        )


def _run_collection_locked(
    curriculum_path: Path,
    root: Path,
    api_key: str,
    transport: Callable[[dict[str, Any], str], dict[str, Any]],
    provider_lookup: Callable[[str, str], dict[str, Any]],
    max_new_calls: int | None,
    expected_curriculum_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(expected_curriculum_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_curriculum_sha256
    ):
        raise ValueError("A lowercase expected curriculum SHA-256 lock is required for teacher calls")
    if file_sha256(curriculum_path) != expected_curriculum_sha256:
        raise RuntimeError("Externally pinned curriculum SHA-256 does not match")
    document, splits = load_curriculum(curriculum_path)
    train = splits["train"]
    manifest = _manifest(curriculum_path, document, train)
    forbidden = _forbidden_values(document, splits)
    root.mkdir(parents=True, exist_ok=True)
    old_manifest = read_json(root / "manifest.json")
    if old_manifest is not None and old_manifest != manifest:
        raise RuntimeError("Run manifest/hash mismatch; refusing changed curriculum, request, or grader")
    write_once(root / "manifest.json", manifest)
    _reconcile_inflight(root)

    canonical_preflight = [_canonical_receipt(row) for row in train]
    if not all(all(receipt.values()) for receipt in canonical_preflight):
        raise RuntimeError("Canonical prompt/oracle/mutation preflight failed")

    # Re-audit every planned first-round request before any paid call.
    for start in range(0, len(train), PACKET_SIZE):
        packet = train[start:start + PACKET_SIZE]
        audit_request(make_request(packet, False), packet, forbidden)
    new_calls = 0
    stop_reason: str | None = None
    try:
        for packet_index, start in enumerate(range(0, len(train), PACKET_SIZE)):
            packet = train[start:start + PACKET_SIZE]
            tag = f"packet_{packet_index:03d}_attempt_0"
            _, attempt_path = _attempt_paths(root, tag)
            existed = attempt_path.exists()
            if not existed and max_new_calls is not None and new_calls >= max_new_calls:
                raise SafePause("safe_pause_before_next_request")
            first = _execute_attempt(
                root, tag, packet_index, 0, packet, False, api_key, transport, provider_lookup, forbidden
            )
            if not attempt_path.exists():  # defensive; normally impossible
                raise RuntimeError("Attempt did not persist")
            if first["model"] != MODEL or not first.get("provider"):
                raise CollectionStopped("Teacher identity receipt failed")
            if first["cost_usd"] is None:
                raise CollectionStopped("Completed response lacks valid client-reported cost")
            if not existed:
                new_calls += 1
            failed_ids = {decision["id"] for decision in first["decisions"] if not decision["accepted"]}
            if not failed_ids:
                continue
            retry_rows = [row for row in packet if row["id"] in failed_ids]
            retry_tag = f"packet_{packet_index:03d}_attempt_1"
            _, retry_path = _attempt_paths(root, retry_tag)
            retry_existed = retry_path.exists()
            if not retry_existed and max_new_calls is not None and new_calls >= max_new_calls:
                raise SafePause("safe_pause_before_retry_request")
            _execute_attempt(
                root, retry_tag, packet_index, 1, retry_rows, True, api_key, transport, provider_lookup, forbidden
            )
            if not retry_path.exists():
                raise RuntimeError("Retry did not persist")
            if not retry_existed:
                new_calls += 1
    except (CollectionStopped, SafePause) as exc:
        stop_reason = str(exc)

    result = _summarize(root, manifest, train, forbidden, stop_reason)
    result_path = root / RESULT_NAME
    existing_result = read_json(result_path)
    if existing_result is not None and existing_result.get("training_release") is True and existing_result != result:
        raise RuntimeError("Refusing to overwrite a released teacher-results artifact")
    atomic_json(result_path, result)
    atomic_text(root / (RESULT_NAME + ".sha256"), file_sha256(result_path) + "\n")
    return result


def validate_only(curriculum_path: Path, root: Path | None = None) -> dict[str, Any]:
    document, splits = load_curriculum(curriculum_path)
    manifest = _manifest(curriculum_path, document, splits["train"])
    forbidden = _forbidden_values(document, splits)
    for start in range(0, len(splits["train"]), PACKET_SIZE):
        packet = splits["train"][start:start + PACKET_SIZE]
        audit_request(make_request(packet, False), packet, forbidden)
        audit_request(make_request(packet, True), packet, forbidden)
    canonical_preflight = [_canonical_receipt(row) for row in splits["train"]]
    if not all(all(receipt.values()) for receipt in canonical_preflight):
        raise RuntimeError("Canonical prompt/oracle/mutation preflight failed")
    existing = None
    if root is not None and root.exists():
        with exclusive_run_lock(root):
            existing = validate_existing(root, manifest, splits["train"], forbidden)
    return {"curriculum_valid": True, "manifest": manifest, "existing_result": existing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs" / "STATS_DIVERSE_CURRICULUM.json")
    parser.add_argument("--root", type=Path, help="Separate work root for immutable paid-call evidence")
    parser.add_argument("--expected-curriculum-sha256",
                        help="Externally pinned whole-file curriculum hash; required with --run")
    parser.add_argument("--run", action="store_true", help="Enable bounded teacher calls")
    args = parser.parse_args()
    if args.run:
        if args.root is None:
            parser.error("--run requires --root")
        if args.expected_curriculum_sha256 is None:
            parser.error("--run requires --expected-curriculum-sha256")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is required only with --run")
        result = run_collection(
            args.curriculum,
            args.root,
            key,
            expected_curriculum_sha256=args.expected_curriculum_sha256,
        )
        result_sha256 = file_sha256(args.root / RESULT_NAME)
        print(json.dumps({
            "training_release": result["training_release"],
            "counts": result["counts"],
            "usage": result["usage"],
            "stop_reason": result["stop_reason"],
            "teacher_results_sha256": result_sha256,
            "teacher_evidence_tree_sha256":
                result["receipts"]["teacher_evidence_tree"]["tree_sha256"],
        }, ensure_ascii=False))
        if not result["training_release"]:
            raise SystemExit(2)
    else:
        result = validate_only(args.curriculum, args.root)
        print(json.dumps({"curriculum_valid": True, "curriculum_sha256": result["manifest"]["curriculum_sha256"],
                          "existing_training_release": (result["existing_result"] or {}).get("training_release")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
