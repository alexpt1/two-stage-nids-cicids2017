import os
import torch
from torch import optim
from tqdm import tqdm

from data_utils import load_nsl_kdd_raw, preprocess_nsl_kdd
from vae_model import VAE, vae_loss_function


def train_vae_on_nsl_kdd(
    data_dir: str = "../data/nsl_kdd",
    model_save_path: str = "../models/vae_nsl_kdd.pt",
    num_epochs: int = 20,
    latent_dim: int = 16,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading NSL-KDD data...")
    df = load_nsl_kdd_raw(data_dir)

    print("Preprocessing...")
    train_loader, val_loader, test_loader, input_dim, preprocessor = preprocess_nsl_kdd(df)

    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for (batch_x,) in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl = vae_loss_function(x_recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f}")

        
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

        print(
            f"  Val recon error mean - normal: {normal_err.mean():.4f}, "
            f"attack: {attack_err.mean():.4f}"
        )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "preprocessor": preprocessor,
        },
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    
    train_vae_on_nsl_kdd()
