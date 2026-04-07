import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import optim
from tqdm import tqdm
import argparse
import numpy as np
from data_utils import load_cicids2017_raw, preprocess_cicids2017
from vae_model import VAE, vae_loss_function


def train_vae_on_cicids2017(
    data_dir: str = "data/cicids2017",
    outputs_root: str = "outputs",
    run_id: str | None = None,
    num_epochs: int = 50,
    latent_dim: int = 32,
    hidden_dims: tuple = (256, 128),
    beta_max: float = 1.0,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CICIDS2017 data...")
    df_monday, df_attacks = load_cicids2017_raw(data_dir)

    print("Preprocessing...")
    train_loader, val_loader, _test_loader, input_dim, scaler, feature_cols, feature_meta, split_info = preprocess_cicids2017(
        df_monday,
        df_attacks,
    )
    val_labels_raw = split_info["val_labels_raw"]

    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.to(device)

    initial_lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    print(f"Optimizer: Adam(lr={initial_lr}) + ReduceLROnPlateau(mode=min, factor=0.5, patience=5)")

    if run_id is None:
        run_id = "vae_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = run_dir / "model.pt"
    final_model_path = run_dir / "model_final.pt"
    config_path     = run_dir / "config.json"
    log_path        = run_dir / "train_log.json"
    epoch_logs = []

    early_stop_patience = 10
    best_val_normal = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    print(f"Early stopping: patience={early_stop_patience} on val_recon_error_mean_normal")

    for epoch in range(1, num_epochs + 1):
        beta = min(beta_max, beta_max * epoch / 10)
        model.train()
        train_losses  = []
        recon_losses  = []
        kl_losses     = []
        grad_norms    = []

        for (batch_x,) in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl = vae_loss_function(x_recon, batch_x, mu, logvar, beta=beta)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl.item())
            grad_norms.append(float(grad_norm))

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_recon_loss = sum(recon_losses) / len(recon_losses)
        avg_kl_loss    = sum(kl_losses)    / len(kl_losses)
        kl_recon_ratio = avg_kl_loss / avg_recon_loss if avg_recon_loss != 0 else 0.0
        mean_grad_norm = float(np.mean(grad_norms))
        max_grad_norm  = float(np.max(grad_norms))
        pct_clipped    = float(np.mean([g > 1.0 for g in grad_norms]) * 100)

        print(
            f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f} "
            f"(beta={beta:.3f}, kl/recon={kl_recon_ratio:.4f}) "
            f"| grad_norm mean={mean_grad_norm:.3f} max={max_grad_norm:.3f} clipped={pct_clipped:.1f}%"
        )

        model.eval()
        recon_errors = []
        labels       = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                x_recon, mu, logvar = model(x_val)
                err = ((x_recon - x_val) ** 2).mean(dim=1).cpu()
                recon_errors.extend(err.numpy())
                labels.extend(y_val.numpy())

        recon_errors = np.array(recon_errors)
        labels       = np.array(labels)

        normal_errors = recon_errors[labels == 0]
        attack_errors = recon_errors[labels == 1]
        normal_mean   = float(normal_errors.mean())
        normal_median = float(np.median(normal_errors))
        attack_mean   = float(attack_errors.mean())
        attack_median = float(np.median(attack_errors))

        per_category_stats = {}
        unique_categories = sorted(set(val_labels_raw))
        for cat in unique_categories:
            cat_mask = val_labels_raw == cat
            if cat_mask.sum() == 0:
                continue
            cat_errors = recon_errors[cat_mask]
            per_category_stats[cat] = {
                "count":  int(cat_mask.sum()),
                "mean":   float(cat_errors.mean()),
                "median": float(np.median(cat_errors)),
            }

        print(
            f"  Val recon error - normal: mean={normal_mean:.4f} median={normal_median:.4f} "
            f"(n={len(normal_errors)}) | "
            f"attack: mean={attack_mean:.4f} median={attack_median:.4f} "
            f"(n={len(attack_errors)})"
        )
        print(f"  Val per-category breakdown:")
        for cat, stats in sorted(per_category_stats.items(), key=lambda kv: -kv[1]["mean"]):
            print(f"    {cat:30s} n={stats['count']:6d}  "
                  f"mean={stats['mean']:.4f}  median={stats['median']:.4f}")

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(normal_mean)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  LR reduced: {lr_before:.2e} -> {lr_after:.2e}")
        else:
            print(f"  LR: {lr_after:.2e}")

        improved = normal_mean < best_val_normal
        if improved:
            best_val_normal = normal_mean
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim":        input_dim,
                    "latent_dim":       latent_dim,
                    "hidden_dims":      list(hidden_dims),
                    "scaler":           scaler,
                    "feature_cols":     feature_cols,
                    "feature_meta":     feature_meta,
                    "epoch":            epoch,
                    "val_normal_mean":  normal_mean,
                },
                model_save_path,
            )
            print(f"  New best: val_normal_mean={normal_mean:.4f} (epoch {epoch}) — saved to {model_save_path.name}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{early_stop_patience} epochs "
                  f"(best={best_val_normal:.4f} at epoch {best_epoch})")

        epoch_logs.append({
            "epoch":                         epoch,
            "lr":                            float(lr_after),
            "train_loss":                    float(avg_train_loss),
            "recon_loss":                    float(avg_recon_loss),
            "kl_loss":                       float(avg_kl_loss),
            "kl_recon_ratio":                float(kl_recon_ratio),
            "beta":                          float(beta),
            "grad_norm_mean":                mean_grad_norm,
            "grad_norm_max":                 max_grad_norm,
            "grad_norm_pct_clipped":         pct_clipped,
            "val_normal_count":              int(len(normal_errors)),
            "val_attack_count":              int(len(attack_errors)),
            "val_recon_error_mean_normal":   normal_mean,
            "val_recon_error_median_normal": normal_median,
            "val_recon_error_mean_attack":   attack_mean,
            "val_recon_error_median_attack": attack_median,
            "val_per_category":              per_category_stats,
        })

        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement for {early_stop_patience} epochs). "
                  f"Best epoch: {best_epoch} with val_normal_mean={best_val_normal:.4f}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim":        input_dim,
            "latent_dim":       latent_dim,
            "hidden_dims":      list(hidden_dims),
            "scaler":           scaler,
            "feature_cols":     feature_cols,
            "feature_meta":     feature_meta,
        },
        final_model_path,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id":       run_id,
                "created_at":   datetime.now(timezone.utc).isoformat(),
                "data_dir":     str(data_dir),
                "num_epochs":   num_epochs,
                "latent_dim":   latent_dim,
                "hidden_dims":  list(hidden_dims),
                "device":       device,
                "input_dim":    int(input_dim),
                "dataset":      "cicids2017",
                "initial_lr":   initial_lr,
                "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=5)",
                "beta_max":     beta_max,
                "beta_schedule": f"min({beta_max}, {beta_max} * epoch / 10)",
                "early_stop_patience": early_stop_patience,
                "best_epoch":   best_epoch,
                "best_val_normal_mean": best_val_normal,
                "epochs_completed": len(epoch_logs),
                "feature_audit": feature_meta["audit_report"],
            },
            f,
            indent=2,
        )

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(epoch_logs, f, indent=2)

    print(f"Run directory   : {run_dir}")
    print(f"Best model      : {model_save_path} (epoch {best_epoch}, val_normal_mean={best_val_normal:.4f})")
    print(f"Final model     : {final_model_path} (epoch {len(epoch_logs)})")
    print(f"Config saved    : {config_path}")
    print(f"Training log    : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on CICIDS2017.")
    parser.add_argument("--data-dir",     default="data/cicids2017")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--num-epochs",   type=int,   default=50)
    parser.add_argument("--latent-dim",   type=int,   default=32)
    parser.add_argument("--hidden-dims",  nargs="+",  type=int, default=[256, 128])
    parser.add_argument("--beta-max",     type=float, default=1.0)
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    train_vae_on_cicids2017(
        data_dir=args.data_dir,
        outputs_root=args.outputs_root,
        num_epochs=args.num_epochs,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        beta_max=args.beta_max,
        device=args.device,
    )