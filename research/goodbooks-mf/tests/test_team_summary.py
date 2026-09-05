import json
from pathlib import Path

from build_team_summary import combine_results, write_chart, write_table


ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def test_team_summary_combines_all_planned_models_and_marks_completion(tmp_path):
    ziqi = _load("ziqi_unified_test_metrics.json")
    yutao = _load("yutao_unified_test_metrics.json")
    svdpp_row = dict(ziqi["models"]["funksvd"])
    svdpp_row.update(model="svdpp", best_validation_metric=0.8)
    ricky = {
        "dataset_version": ziqi["dataset_version"],
        "seed": ziqi["seed"],
        "data_counts": ziqi["data_counts"],
        "ranking_protocol": ziqi["ranking_protocol"],
        "results": [svdpp_row],
    }
    payload = combine_results(
        ziqi,
        yutao,
        ricky,
        _load("validation_selection.json"),
    )

    assert payload["included_models"] == [
        "basic_mf",
        "funksvd",
        "als",
        "nmf",
        "svdpp",
        "bias_aware_als",
    ]
    assert payload["pending_models"] == []
    assert payload["status"] == "complete"
    assert next(row for row in payload["results"] if row["model"] == "svdpp")[
        "owner"
    ] == "Ricky"
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
