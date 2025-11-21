Unsupervised Anomaly Detection in Network Traffic Using Variational Autoencoders (VAEs)

This project implements an unsupervised anomaly detection system for network traffic using Variational Autoencoders (VAEs). The goal is to model normal behaviour in network flow data and identify anomalies using reconstruction error — helping detect potential intrusions, novel attacks, and abnormal traffic patterns.

📌 Project Overview

Modern networks generate massive volumes of traffic, making manual inspection and traditional signature-based intrusion detection ineffective. This project explores deep generative modelling (VAEs) as a way to automatically learn normal behaviour without labelled attack data.

The VAE is trained only on normal network traffic. During testing, high reconstruction error indicates an anomaly.

📂 Datasets Used

Open-source network traffic datasets:

NSL-KDD — 42 features, improved version of KDD’99

CICDDoS2019 — modern dataset with 80 flow-based features

Additional datasets (e.g., Kaggle) may be used for cross-environment validation


All datasets were cleaned, normalised, and pre-processed. Large CSVs are intentionally excluded from the repository.

🧠 Model Architecture

Implemented in Python 3.11 with PyTorch

Encoder → latent space → decoder

Reconstruction loss + KL-divergence

Trained only on normal traffic

Anomaly score = reconstruction error


⚙️ Tools & Libraries

PyTorch

NumPy

Pandas

Matplotlib

Scikit-learn (for baseline comparison, e.g., SVM)


📊 Evaluation

Performance is assessed using:

Reconstruction error analysis

Precision, recall, F1-score

Comparison with baseline ML methods (e.g., SVM)


🚫 What This Project Does Not Cover

Real-time detection, deployment on a live network, or collecting raw traffic data are outside the scope.


📎 Repository Structure
vae-project/
│── src/                # VAE model code
│── data/               # Placeholder for datasets (ignored in Git)
│── notebooks/          # Experiments & visualisations
│── models/             # Saved VAE weights
│── utils/              # Preprocessing helpers
│── README.md
│── requirements.txt
│── .gitignore

📝 License

This project uses only openly available datasets and complies with ethical and data protection guidance provided in the project specification.
