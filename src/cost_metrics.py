import json
import argparse
from pathlib import Path

ATTACK_COSTS = {
    "DoS":   10,
    "Probe":  5,
    "R2L":   20,
    "U2R":   50,
}
FP_COST = 1

def compute_cost_weighted_detection(
    clf_metrics_path: str,
    vae_metrics_path: str,
    attack_costs: dict = None,
    fp_cost: float = FP_COST,
):
    if attack_costs is None:
        attack_costs = ATTACK_COSTS

    with Path(clf_metrics_path).open("r") as f:
        clf = json.load(f)

    with Path(vae_metrics_path).open("r") as f:
        vae = json.load(f)

    cm        = clf["confusion_matrix"]
    classes   = clf["class_names"]
    vae_cm    = vae["confusion_matrix"]

    vae_fp    = int(vae_cm["fp"])
    vae_fn    = int(vae_cm["fn"])
    vae_tp    = int(vae_cm["tp"])

    print("=" * 55)
    print("COST-WEIGHTED DETECTION ANALYSIS")
    print("=" * 55)
    print(f"\nCost weights: {attack_costs}")
    print(f"FP analyst cost: {fp_cost} per alert\n")

    total_fn_cost = 0.0
    cm_matrix = cm if isinstance(cm, list) else []

    for i, cls in enumerate(classes):
        cost = attack_costs.get(cls, 1)
        if cm_matrix:
            row = cm_matrix[i]
            fn_count = sum(row) - row[i]
        else:
            fn_count = 0
        cls_cost = fn_count * cost
        total_fn_cost += cls_cost
        print(f"  {cls:10s} | cost={cost:3d} | FN={fn_count:5d} | cost={cls_cost:8.0f}")

    total_fp_cost = vae_fp * fp_cost
    total_cost    = total_fn_cost + total_fp_cost

    print(f"\n  FP analyst load  | cost={fp_cost:3.0f} | FP={vae_fp:5d} | "
          f"cost={total_fp_cost:8.0f}")
    print("-" * 55)
    print(f"  Total security cost C = {total_cost:,.0f}")
    print(f"  FN contribution       = {total_fn_cost:,.0f}  "
          f"({100*total_fn_cost/total_cost:.1f}%)")
    print(f"  FP contribution       = {total_fp_cost:,.0f}  "
          f"({100*total_fp_cost/total_cost:.1f}%)")
    print("=" * 55)

    payload = {
        "total_cost": total_cost,
        "fn_cost": total_fn_cost,
        "fp_cost": total_fp_cost,
        "attack_costs": attack_costs,
        "fp_unit_cost": fp_cost,
        "per_class_fn_cost": {
            cls: (sum(cm_matrix[i]) - cm_matrix[i][i]) * attack_costs.get(cls, 1)
            for i, cls in enumerate(classes)
        },
    }

    out_path = Path(clf_metrics_path).parent / "cost_metrics.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cost-weighted detection metrics.")
    parser.add_argument("--clf-metrics",  required=True,
                        help="Path to clf run metrics.json")
    parser.add_argument("--vae-metrics",  required=True,
                        help="Path to VAE run metrics.json")
    parser.add_argument("--fp-cost",      type=float, default=1.0)
    args = parser.parse_args()

    compute_cost_weighted_detection(
        clf_metrics_path=args.clf_metrics,
        vae_metrics_path=args.vae_metrics,
        fp_cost=args.fp_cost,
    )