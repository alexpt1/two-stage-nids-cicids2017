import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_nsl_kdd_raw, preprocess_nsl_kdd
from vae_model import VAE
import sklearn
from torch.serialization import add_safe_globals


def load_model(model_path, device):
    
    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler
    ])

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def get_reconstruction_errors(model, loader, device):
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            err = ((x_recon - x) ** 2).mean(dim=1).cpu().numpy()
            errors.extend(err)
            labels.extend(y.numpy())

    return np.array(errors), np.array(labels)


def plot_distribution(errors, labels):
    normal = errors[labels == 0]
    attack = errors[labels == 1]

    threshold = 0.066158
    
    
    bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 80)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Reconstruction Error Distribution - NSL-KDD")

    plt.hist(normal, bins=bins, alpha=0.6, density=True, label="Normal")
    plt.hist(attack, bins=bins, alpha=0.6, density=True, label="Attack")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.xscale("log")
    plt.xlabel("Reconstruction Error (log scale)")
    plt.ylabel("Density")
    plt.title("Distribution of Reconstruction Error\nNormal vs Attack (Log-Scaled Bins)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    print(f"Using : {device}")
        
    df = load_nsl_kdd_raw("../data/nsl_kdd")
    train_loader, val_loader, test_loader, input_dim, preprocessor = preprocess_nsl_kdd(df)

    model = load_model("../models/vae_nsl_kdd.pt", device)

    errors, labels = get_reconstruction_errors(model, test_loader, device)

    plot_distribution(errors, labels)