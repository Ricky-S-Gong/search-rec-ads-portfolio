import json
from pathlib import Path

from build_team_summary import combine_results, write_chart, write_table


ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def test_team_summary_combines_shared_results_and_records_pending_svdpp(tmp_path):
    payload = combine_results(
        _load("ziqi_unified_test_metrics.json"),
        _load("yutao_unified_test_metrics.json"),
        _load("validation_selection.json"),
    )

    assert payload["included_models"] == [
        "basic_mf",
        "funksvd",
        "als",
        "nmf",
        "bias_aware_als",
    ]
    assert payload["pending_models"] == ["svdpp"]
    assert {row["evaluated_rating_count"] for row in payload["results"]} == {17168}
    assert {row["evaluated_ranking_users"] for row in payload["results"]} == {4765}
    assert payload["ranking_protocol"]["candidate_policy"] == (
        "full_train_catalog_excluding_seen"
    )

    table_path = tmp_path / "comparison.csv"
    chart_path = tmp_path / "comparison.png"
    write_table(payload, table_path)
    write_chart(payload, chart_path)
    assert table_path.read_text(encoding="utf-8").startswith(
        "model,model_label,owner,model_role,"
    )
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
