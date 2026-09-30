from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARM_ORDER = ["hybrid_muon", "scion_c1_s1", "afmoun_c3_s0p5"]


def arm_from_name(name: str) -> str:
    for arm in ARM_ORDER:
        if arm in name:
            return arm
    return name


def short_arm(arm: str) -> str:
    return {
        "hybrid_muon": "Hybrid",
        "scion_c1_s1": "SCION",
        "afmoun_c3_s0p5": "AF-Muon",
    }.get(arm, arm)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def finite_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def last_phase(rows: list[dict], phase: str) -> dict | None:
    matches = [r for r in rows if r.get("phase") == phase]
    return matches[-1] if matches else None


def first_phase(rows: list[dict], phase: str) -> dict | None:
    matches = [r for r in rows if r.get("phase") == phase]
    return matches[0] if matches else None


def at_or_before(rows: list[dict], phase: str, step: int) -> dict | None:
    matches = [r for r in rows if r.get("phase") == phase and int(r.get("step", -1)) <= int(step)]
    return matches[-1] if matches else None


def slope(first: dict | None, last: dict | None, key: str) -> float:
    if first is None or last is None:
        return float("nan")
    x0 = finite_or_nan(first.get("target_tokens_seen", first.get("tokens_seen", 0))) / 1e6
    x1 = finite_or_nan(last.get("target_tokens_seen", last.get("tokens_seen", 0))) / 1e6
    y0 = finite_or_nan(first.get(key))
    y1 = finite_or_nan(last.get(key))
    if not all(math.isfinite(v) for v in [x0, x1, y0, y1]) or x1 == x0:
        return float("nan")
    return (y1 - y0) / (x1 - x0)


def fmt(value: float, digits: int = 4) -> str:
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def summarize_run(root: Path) -> list[dict]:
    out = []
    for metrics in sorted(root.glob("*/metrics.jsonl")):
        run_dir = metrics.parent
        rows = load_rows(metrics)
        config = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        arm = arm_from_name(run_dir.name)
        first_train = first_phase(rows, "train")
        last_train = last_phase(rows, "train")
        first_eval = first_phase(rows, "eval")
        last_eval = last_phase(rows, "eval")
        first_diag = first_phase(rows, "diagnostic")
        last_diag = last_phase(rows, "diagnostic")
        nonfinite = 0
        for row in rows:
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    nonfinite += 1

        row = {
            "run_root": str(root),
            "run_name": run_dir.name,
            "arm": arm,
            "arm_label": short_arm(arm),
            "rows": len(rows),
            "nonfinite_count": nonfinite,
            "param_count": config.get("param_count"),
            "sharing": config.get("sharing", "full"),
            "target_tokens_per_step": config.get("target_tokens_per_step"),
            "total_raw_tokens_per_step": config.get("total_raw_tokens_per_step"),
            "matrix_wd": config.get("weight_decay_protocol", {}).get("matrix_weight_decay_all_arms"),
            "hybrid_aux_wd": config.get("weight_decay_protocol", {}).get("hybrid_aux_weight_decay"),
            "sign_aux_wd": config.get("weight_decay_protocol", {}).get("sign_tied_weight_decay"),
            "afmoun_aux_wd": config.get("weight_decay_protocol", {}).get("afmoun_tied_weight_decay"),
            "final_step": None if last_eval is None else last_eval.get("step"),
            "final_target_tokens": None if last_eval is None else last_eval.get("target_tokens_seen", last_eval.get("tokens_seen")),
            "final_total_raw_tokens": None if last_eval is None else last_eval.get("total_raw_tokens_seen"),
            "first_eval_loss": None if first_eval is None else first_eval.get("eval_loss"),
            "final_eval_loss": None if last_eval is None else last_eval.get("eval_loss"),
            "final_eval_ppl": None if last_eval is None else last_eval.get("eval_ppl"),
            "eval_loss_delta": None
            if first_eval is None or last_eval is None
            else finite_or_nan(last_eval.get("eval_loss")) - finite_or_nan(first_eval.get("eval_loss")),
            "last_train_loss": None if last_train is None else last_train.get("loss"),
            "last_grad_norm": None if last_train is None else last_train.get("grad_norm_preclip"),
            "zero_encoder_loss_delta": None if last_eval is None else last_eval.get("zero_encoder_loss_delta"),
            "shuffle_encoder_loss_delta": None if last_eval is None else last_eval.get("shuffle_encoder_loss_delta"),
            "shared_split_grad_rel_error": None if last_diag is None else last_diag.get("shared_split_grad_rel_error"),
            "enc_lookup_frac": None if last_diag is None else last_diag.get("enc_lookup_grad_nonzero_row_frac"),
            "dec_lookup_frac": None if last_diag is None else last_diag.get("dec_lookup_grad_nonzero_row_frac"),
            "out_head_frac": None if last_diag is None else last_diag.get("out_head_grad_nonzero_row_frac"),
            "enc_grad_fro": None if last_diag is None else last_diag.get("enc_lookup_grad_fro"),
            "dec_grad_fro": None if last_diag is None else last_diag.get("dec_lookup_grad_fro"),
            "out_grad_fro": None if last_diag is None else last_diag.get("out_head_grad_fro"),
            "shared_grad_fro": None if last_diag is None else last_diag.get("shared_grad_fro"),
            "enc_out_cos_active": None if last_diag is None else last_diag.get("enc_out_grad_cos_active_rows"),
            "dec_out_cos_active": None if last_diag is None else last_diag.get("dec_out_grad_cos_active_rows"),
            "enc_dec_cos_union": None if last_diag is None else last_diag.get("enc_dec_grad_cos_union"),
            "shared_rms_first": None if first_diag is None else first_diag.get("shared_rms"),
            "shared_rms_last": None if last_diag is None else last_diag.get("shared_rms"),
            "shared_max_last": None if last_diag is None else last_diag.get("shared_max_abs"),
            "shared_rms_slope_per_mtok": slope(first_diag, last_diag, "shared_rms"),
            "shared_max_slope_per_mtok": slope(first_diag, last_diag, "shared_max_abs"),
            "encoder_embed_rms_last": None if last_diag is None else last_diag.get("encoder_embed_rms"),
            "decoder_embed_rms_last": None if last_diag is None else last_diag.get("decoder_embed_rms"),
            "output_weight_rms_last": None if last_diag is None else last_diag.get("output_weight_rms"),
            "encoder_embed_max_last": None if last_diag is None else last_diag.get("encoder_embed_max_abs"),
            "decoder_embed_max_last": None if last_diag is None else last_diag.get("decoder_embed_max_abs"),
            "output_weight_max_last": None if last_diag is None else last_diag.get("output_weight_max_abs"),
            "seconds_last_eval": None if last_eval is None else last_eval.get("seconds"),
        }
        row["dec_over_enc_grad_fro"] = (
            finite_or_nan(row["dec_grad_fro"]) / finite_or_nan(row["enc_grad_fro"])
            if finite_or_nan(row["enc_grad_fro"]) != 0
            else float("nan")
        )
        row["out_over_dec_grad_fro"] = (
            finite_or_nan(row["out_grad_fro"]) / finite_or_nan(row["dec_grad_fro"])
            if finite_or_nan(row["dec_grad_fro"]) != 0
            else float("nan")
        )
        out.append(row)
    return sorted(out, key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)


def print_table(rows: list[dict], title: str, keys: list[tuple[str, str]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    headers = ["arm"] + [h for h, _ in keys]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        values = [row["arm_label"]]
        for _, key in keys:
            value = row.get(key)
            if isinstance(value, (float, int)):
                values.append(fmt(value))
            else:
                values.append(str(value))
        print(" | ".join(values))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize three-way shared-vocab t5test runs.")
    p.add_argument("--run-roots", nargs="+", required=True)
    p.add_argument("--csv-out", default="")
    args = p.parse_args()

    all_rows = []
    for raw in args.run_roots:
        root = Path(raw)
        rows = summarize_run(root)
        all_rows.extend(rows)
        print("\n" + "=" * 100)
        print(root)
        print("=" * 100)
        if not rows:
            print("No metrics.jsonl files found.")
            continue
        print_table(
            rows,
            "Performance",
            [
                ("sharing", "sharing"),
                ("params", "param_count"),
                ("step", "final_step"),
                ("target_M", "final_target_tokens"),
                ("eval_loss", "final_eval_loss"),
                ("eval_ppl", "final_eval_ppl"),
                ("loss_delta", "eval_loss_delta"),
                ("train_loss", "last_train_loss"),
                ("grad_norm", "last_grad_norm"),
            ],
        )
        print_table(
            rows,
            "Vocabulary Table Growth",
            [
                ("rms_first", "shared_rms_first"),
                ("rms_last", "shared_rms_last"),
                ("max_last", "shared_max_last"),
                ("rms_slope/M", "shared_rms_slope_per_mtok"),
                ("max_slope/M", "shared_max_slope_per_mtok"),
                ("enc_rms", "encoder_embed_rms_last"),
                ("dec_rms", "decoder_embed_rms_last"),
                ("out_rms", "output_weight_rms_last"),
                ("enc_max", "encoder_embed_max_last"),
                ("dec_max", "decoder_embed_max_last"),
                ("out_max", "output_weight_max_last"),
            ],
        )
        print_table(
            rows,
            "Role Geometry",
            [
                ("split_rel", "shared_split_grad_rel_error"),
                ("enc_frac", "enc_lookup_frac"),
                ("dec_frac", "dec_lookup_frac"),
                ("out_frac", "out_head_frac"),
                ("dec/enc", "dec_over_enc_grad_fro"),
                ("out/dec", "out_over_dec_grad_fro"),
                ("enc_out_cos", "enc_out_cos_active"),
                ("dec_out_cos", "dec_out_cos_active"),
            ],
        )
        print_table(
            rows,
            "Encoder Dependence",
            [
                ("zero_delta", "zero_encoder_loss_delta"),
                ("shuffle_delta", "shuffle_encoder_loss_delta"),
            ],
        )
        print_table(
            rows,
            "WD Protocol",
            [
                ("matrix_wd", "matrix_wd"),
                ("hybrid_aux_wd", "hybrid_aux_wd"),
                ("sign_aux_wd", "sign_aux_wd"),
                ("afmoun_aux_wd", "afmoun_aux_wd"),
                ("nonfinite", "nonfinite_count"),
            ],
        )

    if args.csv_out and all_rows:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in all_rows for key in row})
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote CSV: {out_path}")


if __name__ == "__main__":
    main()
