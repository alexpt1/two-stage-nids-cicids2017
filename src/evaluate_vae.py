import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import sklearn
from torch.serialization import add_safe_globals

from data_utils import load_nsl_kdd_raw, preprocess_nsl_kdd
from vae_model import VAE


def evaluate_vae_nsl_kdd(
    data_dir: str = "../data/nsl_kdd",
    model_path: str = "../models/vae_nsl_kdd.pt",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    
    df = load_nsl_kdd_raw(data_dir)
    train_loader, val_loader, test_loader, input_dim, preprocessor = preprocess_nsl_kdd(df)

    
    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler
    ])

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)    
    latent_dim = checkpoint.get("latent_dim", 16)
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    
    val_errors = []
    val_labels = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            x_recon, mu, logvar = model(x_val)
            err = ((x_recon - x_val) ** 2).mean(dim=1).cpu().numpy()
            val_errors.extend(err)
            val_labels.extend(y_val.numpy())

    val_errors = np.array(val_errors)
    val_labels = np.array(val_labels)

    
    normal_val_errors = val_errors[val_labels == 0]
    threshold = np.percentile(normal_val_errors, 95)
    print(f"Chosen anomaly threshold (95th percentile of normal val errors): {threshold:.6f}")

    
    test_errors = []
    test_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            x_recon, mu, logvar = model(x_test)
            err = ((x_recon - x_test) ** 2).mean(dim=1).cpu().numpy()
            test_errors.extend(err)
            test_labels.extend(y_test.numpy())

    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    y_pred = (test_errors > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average="binary", pos_label=1
    )
    auc = roc_auc_score(test_labels, test_errors)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")


if __name__ == "__main__":
    evaluate_vae_nsl_kdd()
