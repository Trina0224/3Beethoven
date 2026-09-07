"""Shared, conservative teacher-envelope recovery; never rewrites expressions."""
import json


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_answers(raw, expected_ids=None):
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    stripped = raw.strip()
    payload, end = decoder.raw_decode(stripped)
    suffix = stripped[end:].strip()
    repairs = []
    if suffix:
        if suffix != "}":
            raise ValueError("Unrecognized trailing content")
        repairs.append("removed_redundant_trailing_closing_brace")
    if not isinstance(payload, dict):
        raise ValueError("Envelope must be an object")
    nested = payload.get("required_response")
    nested_answers = isinstance(nested, dict) and "answers" in nested
    if "answers" in payload and nested_answers:
        raise ValueError("Ambiguous root and nested answers")
    if nested_answers:
        payload = nested
        repairs.append("extracted_echoed_required_response")
    items = payload.get("answers")
    if not isinstance(items, list) or not items:
        raise ValueError("answers must be a nonempty list")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"question_id", "expression"}:
            raise ValueError("Each answer must contain only question_id and expression")
        qid, expression = item["question_id"], item["expression"]
        if not isinstance(qid, str) or not isinstance(expression, str) or not expression.strip():
            raise ValueError("Answer fields must be nonempty strings")
        if qid in seen or (expected_ids is not None and qid not in expected_ids):
            raise ValueError("Duplicate or unexpected question_id")
        if "<" in expression or ">" in expression:
            raise ValueError("Placeholder or unsupported expression")
        seen.add(qid)
    return items, ";".join(repairs) or None
