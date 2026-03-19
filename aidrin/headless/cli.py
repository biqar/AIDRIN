import argparse
import json
import sys
from typing import List, Optional

from .api import list_available_metrics, run_batch_metrics, run_data_quality, run_metric
from .config import HeadlessConfig


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _dump_result(result: object) -> None:
    sys.stdout.write(json.dumps(result, indent=2))
    sys.stdout.write("\n")


def _summarize_data_quality(result: dict) -> None:
    """Print a compact summary of data quality results."""
    # Completeness
    comp = result.get("completeness", {})
    overall_comp = comp.get("Overall Completeness", "N/A")
    scores = comp.get("Completeness scores", {})
    n_features = len(scores)
    if scores:
        vals = list(scores.values())
        min_comp = min(vals)
        incomplete = sum(1 for v in vals if v < 1.0)
    else:
        min_comp = "N/A"
        incomplete = 0

    # Duplicity
    dup = result.get("duplicity", {})
    dup_scores = dup.get("Duplicity scores", {})
    dup_ratio = dup_scores.get("Overall duplicity of the dataset", "N/A")

    # Outliers
    out = result.get("outliers", {})
    out_scores = out.get("Outlier scores", {})
    overall_outlier = out_scores.get("Overall outlier score", "N/A")
    feature_outliers = {k: v for k, v in out_scores.items() if k != "Overall outlier score"}
    if feature_outliers:
        max_outlier = max(feature_outliers.values())
        high_outlier = sum(1 for v in feature_outliers.values() if v > 0.05)
    else:
        max_outlier = "N/A"
        high_outlier = 0

    print(f"Data Quality Summary ({n_features} features)")
    print(f"{'='*45}")
    print(f"Completeness:  {_fmt(overall_comp)}  (min: {_fmt(min_comp)}, incomplete: {incomplete}/{n_features})")
    print(f"Duplicity:     {_fmt(dup_ratio)}")
    print(f"Outliers:      {_fmt(overall_outlier)}  (max: {_fmt(max_outlier)}, >5%: {high_outlier}/{n_features})")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _summarize_metric(metric_name: str, result: dict) -> None:
    """Print a compact summary for a single metric result."""
    if metric_name == "completeness":
        overall = result.get("Overall Completeness", "N/A")
        scores = result.get("Completeness scores", {})
        n = len(scores)
        if scores:
            vals = list(scores.values())
            incomplete = sum(1 for v in vals if v < 1.0)
            print(f"Completeness ({n} features): {_fmt(overall)}  (min: {_fmt(min(vals))}, incomplete: {incomplete}/{n})")
        else:
            print(f"Completeness: {_fmt(overall)}")
    elif metric_name == "duplicity":
        dup_scores = result.get("Duplicity scores", {})
        ratio = dup_scores.get("Overall duplicity of the dataset", "N/A")
        print(f"Duplicity: {_fmt(ratio)}")
    elif metric_name == "outliers":
        scores = result.get("Outlier scores", {})
        overall = scores.get("Overall outlier score", "N/A")
        feature_scores = {k: v for k, v in scores.items() if k != "Overall outlier score"}
        n = len(feature_scores)
        if feature_scores:
            mx = max(feature_scores.values())
            high = sum(1 for v in feature_scores.values() if v > 0.05)
            print(f"Outliers ({n} features): {_fmt(overall)}  (max: {_fmt(mx)}, >5%: {high}/{n})")
        else:
            print(f"Outliers: {_fmt(overall)}")
    else:
        # Generic: print top-level keys with scalar values, count dict/list values
        for k, v in result.items():
            if "visualization" in k.lower():
                continue
            if isinstance(v, dict):
                print(f"{k}: ({len(v)} entries)")
            elif isinstance(v, list):
                print(f"{k}: [{len(v)} items]")
            else:
                print(f"{k}: {_fmt(v)}")


def _build_run_kwargs(args: argparse.Namespace) -> dict:
    return {
        "columns": _parse_list(args.columns),
        "target_column": args.target_column,
        "quasi_identifiers": _parse_list(args.quasi_identifiers),
        "sensitive_column": args.sensitive_column,
        "epsilon": args.epsilon,
        "id_column": args.id_column,
        "eval_columns": _parse_list(args.eval_columns),
        "distance_metric": args.distance_metric,
        "cat_columns": _parse_list(args.cat_columns),
        "num_columns": _parse_list(args.num_columns),
        "y_true_column": args.y_true_column,
        "sensitive_attribute_column": args.sensitive_attribute_column,
        "save_images": args.save_images,
        "image_dir": args.image_dir,
        "verbose": getattr(args, "verbose", False),
        "strip_visualizations": getattr(args, "no_viz", False),
    }


def _configure_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file-type", dest="file_type", default=None)
    parser.add_argument("--columns", default=None, help="Comma-separated column list")
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--quasi-identifiers", default=None)
    parser.add_argument("--sensitive-column", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--eval-columns", default=None)
    parser.add_argument("--distance-metric", default=None)
    parser.add_argument("--cat-columns", default=None)
    parser.add_argument("--num-columns", default=None)
    parser.add_argument("--y-true-column", default=None)
    parser.add_argument("--sensitive-attribute-column", default=None)
    parser.add_argument("--save-images", dest="save_images", action="store_true")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress output")
    parser.add_argument("--no-viz", action="store_true", help="Strip visualization data from output", default=True)
    parser.add_argument("--detail", action="store_true", help="Output full per-feature JSON instead of summary", default=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="aidrin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available metrics")
    list_parser.add_argument("--category", default=None)

    run_parser = subparsers.add_parser("run", help="Run a single metric")
    run_parser.add_argument("metric")
    run_parser.add_argument("file_path")
    _configure_common_run_args(run_parser)

    batch_parser = subparsers.add_parser("batch", help="Run metrics from config file (JSON or YAML)")
    batch_parser.add_argument("config_path")
    batch_parser.add_argument("-v", "--verbose", action="store_true", help="Show progress output")
    batch_parser.add_argument("--no-viz", action="store_true", help="Strip visualization data from output")

    # Fast data quality command
    dq_parser = subparsers.add_parser("data-quality", help="Run fast data quality metrics (completeness, duplicity, outliers)")
    dq_parser.add_argument("file_path")
    dq_parser.add_argument("--file-type", dest="file_type", default=None)
    dq_parser.add_argument("-v", "--verbose", action="store_true", help="Show progress output")
    dq_parser.add_argument("--detail", action="store_true", help="Output full per-feature JSON instead of summary")

    args = parser.parse_args()

    try:
        if args.command == "list":
            _dump_result(list_available_metrics(category=args.category))
            return

        if args.command == "run":
            result = run_metric(
                args.metric,
                args.file_path,
                file_type=args.file_type,
                **_build_run_kwargs(args),
            )
            if args.detail:
                _dump_result(result)
            else:
                _summarize_metric(args.metric.strip().lower(), result)
            return

        if args.command == "batch":
            config = HeadlessConfig.from_file(args.config_path)
            result = run_batch_metrics(
                config,
                verbose=args.verbose,
                strip_visualizations=args.no_viz,
            )
            _dump_result(result)
            return

        if args.command == "data-quality":
            result = run_data_quality(
                args.file_path,
                file_type=args.file_type,
                verbose=args.verbose,
                strip_visualizations=True,
            )
            if args.detail:
                _dump_result(result)
            else:
                _summarize_data_quality(result)
            return
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
