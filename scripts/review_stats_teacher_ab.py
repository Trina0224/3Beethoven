"""Post-run diagnostics only. Never changes frozen A/B results or releases training."""
import json
import re
from collections import Counter
from stats_teacher_ab import DOCS, read_json, save_json, judge, digest, spec_for


def review():
    evidence = read_json(DOCS / "STATS_TEACHER_AB_EVIDENCE.json")
    questions = {q["id"]:q for q in read_json(DOCS / "STATS_TEACHER_AB_DATA.json")["questions"]}
    manual = {
        "teacher_ab_affine_poisson_02": "For a=7, m=55, b=6: 49=a^2, so 49*(55+55^2)+6^2+2*7*55*6 equals a^2*(m+m^2)+b^2+2amb. Exact oracle value 155576.",
        "teacher_ab_affine_poisson_05": "For a=3, m=26, b=28: 9=a^2, 6=2a, 784=b^2. Thus 9*(26+26^2)+6*28*26+784 equals a^2*(m+m^2)+2amb+b^2. Exact oracle value 11470.",
    }
    rows = []
    for row in evidence["results.json"]["rows"]:
        disposition = {"id":row["id"], "arm":row["arm"], "family":row["family"],
                       "frozen_classification":row["classification"], "frozen_accepted":row["accepted"],
                       "diagnostic_usable":row["accepted"]}
        if row["arm"] == "A" and row["id"] in manual:
            assert row["classification"] == "equivalence_pending"
            disposition.update(diagnostic_usable=True, review="verified_partial_constant_folding", reason=manual[row["id"]])
        elif row["arm"] == "B":
            if row["classification"] == "non_stop_finish":
                disposition.update(review="truncated_output_not_a_math_failure", reason="finish_reason=length; 400 tokens. Do not invent missing JSON or a final expression.")
            else:
                blocks = re.findall(r"```json\s*(.*?)\s*```", row["raw"], re.S)
                if len(blocks) == 1:
                    checked = judge(blocks[0], questions[row["id"]], "B")
                    disposition.update(review="posthoc_single_fenced_JSON_diagnostic", diagnostic_usable=checked["accepted"], extracted_check=checked)
                    if checked["classification"] == "equivalence_pending" and row["family"] == "affine_poisson":
                        s = spec_for(questions[row["id"]])
                        a,m,b = int(s["scale"]), int(s["mean"]), int(s["offset"])
                        expected = f"({a}*{m}+{b})**2+{a*a}*{m}"
                        assert checked["expression"].replace(" ", "") == expected
                        assert all(checked["intermediate_checks"].values())
                        disposition.update(diagnostic_usable=True, semantic_review="verified_folded_variance_coefficient",
                                           reason="Exactly (a*m+b)^2+a^2*m with only a^2 evaluated; intermediate mean and variance independently agree.")
                else:
                    disposition.update(review="no_unique_complete_JSON_block")
        elif row["classification"] == "non_executable_or_format":
            disposition.update(review="resolved_invalid_target", reason="Contains unevaluated E[...] and X; not an executable substituted numerical expression.")
        elif row["id"] == "teacher_ab_interval_06" and row["arm"] == "A":
            disposition.update(review="format_and_math_error", reason="Extra trailing quote. The visible expression 223-(223-126)/7 also preserves neither center nor scaled half-width; reference is (126+223)/2+(223-126)/14.")
        rows.append(disposition)
    counts = {}
    for arm in ("A", "B"):
        selected = [r for r in rows if r["arm"] == arm]
        counts[arm] = {"diagnostic_usable":sum(r["diagnostic_usable"] for r in selected),
                       "by_family":dict(Counter(r["family"] for r in selected if r["diagnostic_usable"])),
                       "dispositions":dict(Counter(r.get("review", r["frozen_classification"]) for r in selected))}
    return {"role":"posthoc_diagnostic_not_a_replacement_gate", "teacher_calls_added":0,
            "evidence_digest":digest(evidence), "frozen_primary":evidence["results.json"]["arms"],
            "reviewed":counts, "pending_semantic_reviews":0, "training_released":False, "rows":rows}


if __name__ == "__main__":
    result = review()
    destination = DOCS / "STATS_TEACHER_AB_REVIEW.json"
    if destination.exists():
        raise SystemExit("Preserve the existing review; use a new version")
    save_json(destination, result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))
