import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import optim
from tqdm import tqdm
import argparse
from data_utils import load_nsl_kdd_raw, preprocess_nsl_kdd
from vae_model import VAE, vae_loss_function


def train_vae_on_nsl_kdd(
    data_dir: str = "../data/nsl_kdd",
    outputs_root: str = "../outputs",
    run_id: str | None = None,
    num_epochs: int = 20,
    latent_dim: int = 16,
    device: str | None = None,
    reshuffle_all: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading NSL-KDD data...")
    df_train, df_test = load_nsl_kdd_raw(data_dir)

    print("Preprocessing...")
    train_loader, val_loader, _test_loader, input_dim, preprocessor = preprocess_nsl_kdd(
        df_train,
        df_test,
        reshuffle_all=reshuffle_all,
    )

    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = run_dir / "model.pt"
    config_path = run_dir / "config.json"
    log_path = run_dir / "train_log.json"
    epoch_logs = []

    for epoch in range(1, num_epochs + 1):
        beta = min(1.0, epoch / 10)
        model.train()
        train_losses = []
        recon_losses = []
        kl_losses = []
        for (batch_x,) in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl = vae_loss_function(x_recon, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_recon_loss = sum(recon_losses) / len(recon_losses)
        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        kl_recon_ratio = avg_kl_loss / avg_recon_loss if avg_recon_loss != 0 else 0.0
        print(
            f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f} "
            f"(beta={beta:.3f}, kl/recon={kl_recon_ratio:.4f})"
        )

        model.eval()
        recon_errors = []
        labels = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                x_recon, mu, logvar = model(x_val)

                err = ((x_recon - x_val) ** 2).mean(dim=1).cpu()
                recon_errors.extend(err.numpy())
                labels.extend(y_val.numpy())

        import numpy as np

        recon_errors = np.array(recon_errors)
        labels = np.array(labels)

        normal_err = recon_errors[labels == 0]
        attack_err = recon_errors[labels == 1]
        normal_mean = float(normal_err.mean())
        attack_mean = float(attack_err.mean())

        print(
            f"  Val recon error mean - normal: {normal_mean:.4f}, "
            f"attack: {attack_mean:.4f}"
        )
        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "recon_loss": float(avg_recon_loss),
                "kl_loss": float(avg_kl_loss),
                "kl_recon_ratio": float(kl_recon_ratio),
                "beta": float(beta),
                "val_recon_error_mean_normal": normal_mean,
                "val_recon_error_mean_attack": attack_mean,
            }
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "preprocessor": preprocessor,
        },
        model_save_path,
    )
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_dir": str(data_dir),
                "num_epochs": num_epochs,
                "latent_dim": latent_dim,
                "device": device,
                "reshuffle_all": reshuffle_all,
                "input_dim": int(input_dim),
            },
            f,
            indent=2,
        )
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(epoch_logs, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Model saved to {model_save_path}")
    print(f"Config saved to {config_path}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on NSL-KDD.")

    parser.add_argument("--data-dir", default="../data/nsl_kdd")
    parser.add_argument("--outputs-root", default="../outputs")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reshuffle-all", action="store_true")

    args = parser.parse_args()

    train_vae_on_nsl_kdd(
        data_dir=args.data_dir,
        outputs_root=args.outputs_root,
        num_epochs=args.num_epochs,
        latent_dim=args.latent_dim,
        device=args.device,
        reshuffle_all=args.reshuffle_all,
    )