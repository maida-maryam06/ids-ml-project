<div align="center">

<!-- LOGO / BANNER -->
<img src="https://img.shields.io/badge/🛡️_SecureNet_IDS-AI_Powered-00d4ff?style=for-the-badge&labelColor=050a0e" alt="SecureNet IDS" />

# 🛡️ SecureNet IDS — AI-Powered Intrusion Detection System

### Machine Learning Network Traffic Classifier | CIC-IDS2017 | DDoS Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-blue?style=flat-square)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.99%25-brightgreen?style=flat-square)](https://github.com)

<br/>

> **A proof-of-concept ML-based NIDS that detects DDoS attacks from network flow features with 99.99% accuracy using Random Forest on the CIC-IDS2017 dataset.**

<br/>

[🚀 Live Demo](https://share.streamlit.io) · [📓 Notebook](notebook/ids_model.ipynb) · [📄 Report](report.docx) · [🐛 Issues](../../issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Security Analysis](#-security-analysis)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**SecureNet IDS** is a Machine Learning-based **Network Intrusion Detection System (NIDS)** built as a CLO4 Information Security assignment. It trains a **Random Forest classifier** on the **CIC-IDS2017** dataset to automatically classify network traffic as **BENIGN** or **DDoS attack** with over **99.99% accuracy**.

### 🎯 Problem Statement
Traditional firewall-based IDS rely on static rules and known attack signatures — making them blind to novel threats. This project demonstrates how ML can build a **proactive, adaptive** detection layer that identifies DDoS patterns from raw network flow statistics.

### 🏢 Real-World Scenario
> *SecureNet Corp's CISO has tasked the security team with evaluating ML as a tool to augment the existing NIDS and reduce mean time to detection (MTTD) from minutes to milliseconds.*

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

The live Streamlit dashboard lets you:
- 🔄 Switch between **Random Forest** and **Decision Tree** models
- ⚡ Classify live network flow features in real time
- 📊 View accuracy, F1-score, confusion matrices and feature importance charts
- 🔴 Load pre-built **DDoS** or **Benign** traffic demos

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Dual ML Models** | Random Forest (primary) + Decision Tree (baseline comparison) |
| 📡 **Live Classifier** | Input 8 network flow features → instant DDoS / BENIGN prediction |
| 📊 **Full Metrics** | Accuracy, Precision, Recall, F1-Score for both models |
| 🔄 **Model Switcher** | Switch active model mid-session, metrics update instantly |
| 🌐 **REST API** | Flask backend with `/predict`, `/predict_batch`, `/set_model` endpoints |
| 📈 **Auto Charts** | EDA charts, confusion matrices, feature importance auto-generated |
| 🎨 **Dark Dashboard** | Cybersecurity-themed UI (HTML/CSS/JS + Streamlit) |
| 📄 **Full Report** | Word document with methodology, results, security analysis |

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **99.99%** | **99.99%** | **99.99%** | **99.99%** |
| Decision Tree | 99.96% | 99.96% | 99.96% | 99.96% |

> ✅ Evaluated on a **45,143-sample stratified test set** (20% holdout). Results are reproducible by running the Jupyter Notebook.

### Why Such High Accuracy?
DDoS traffic has extremely distinct fingerprints (high forward packet rate, zero backward traffic, small packet sizes) that Random Forest separates from benign flows with near-perfect precision. This is consistent with published research on CIC-IDS2017.

---

## 📦 Dataset

**CIC-IDS2017** — Canadian Institute for Cybersecurity

| Attribute | Value |
|-----------|-------|
| File | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` |
| Classes | `BENIGN`, `DDoS` |
| Features | 78 numerical flow features |
| Attack Types | UDP Flood, TCP SYN Flood, HOIC, LOIC |
| Source | [unb.ca/cic/datasets/ids-2017](https://www.unb.ca/cic/datasets/ids-2017.html) |

> ⚠️ The dataset is not included in this repo (too large). Download it from the link above and place it in `data/`.

---

## 📁 Project Structure

```
CLO4-IDS-ML-Solution/
│
├── 📓 notebook/
│   └── ids_model.ipynb          # Full ML pipeline (EDA → Train → Evaluate)
│
├── 🔧 backend/
│   ├── app.py                   # Flask REST API
│   ├── model_trainer.py         # Standalone training script (no Jupyter needed)
│   ├── requirements.txt         # Backend-only dependencies
│   ├── model.pkl                # Saved Random Forest ← generated by notebook
│   ├── dt_model.pkl             # Saved Decision Tree ← generated by notebook
│   ├── scaler.pkl               # Saved StandardScaler
│   ├── label_encoder.pkl        # Saved LabelEncoder
│   ├── feature_names.pkl        # Feature list
│   └── metrics.pkl              # Accuracy/F1 for both models
│
├── 🌐 frontend/
│   ├── index.html               # Cybersecurity dashboard (HTML/CSS/JS)
│   └── *.png                    # Charts auto-saved by notebook
│
├── 📱 app.py                    # Streamlit dashboard (for cloud deployment)
├── 📦 requirements.txt          # Full project dependencies
├── 📄 report.docx               # Assignment report (Word)
├── 🔒 .gitignore
└── 📖 README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/CLO4-IDS-ML-Solution.git
cd CLO4-IDS-ML-Solution
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and place in `data/`:
```
data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### 4. Train the model

**Option A — Jupyter Notebook:**
```bash
cd notebook
jupyter notebook ids_model.ipynb
# Cell → Run All
```

**Option B — Terminal script:**
```bash
python backend/model_trainer.py
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. (Optional) Run the Flask API
```bash
python backend/app.py
# API at http://127.0.0.1:5000
```

---

## 🔌 API Reference

Base URL: `http://127.0.0.1:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + active model |
| `GET` | `/metrics` | Accuracy/F1 for both models |
| `POST` | `/set_model` | Switch active model |
| `POST` | `/predict` | Classify a single flow |
| `POST` | `/predict_batch` | Classify multiple flows |

### Example — Single Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Flow Duration": 1234567,
    "Total Fwd Packets": 850,
    "Flow Packets/s": 12000.3
  }'
```

**Response:**
```json
{
  "prediction": "DDoS",
  "is_attack": true,
  "confidence": 0.9999,
  "probabilities": { "BENIGN": 0.0001, "DDoS": 0.9999 },
  "model_used": "random_forest"
}
```

### Switch Model
```bash
curl -X POST http://127.0.0.1:5000/set_model \
  -H "Content-Type: application/json" \
  -d '{"model": "decision_tree"}'
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn (Random Forest, Decision Tree) |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Backend API** | Flask, flask-cors |
| **Frontend** | HTML5, CSS3, Vanilla JS |
| **Dashboard** | Streamlit |
| **Persistence** | joblib |
| **Notebook** | Jupyter |

---

## 🔐 Security Analysis

### Why Random Forest?
- **Ensemble averaging** across 100 trees eliminates overfitting on noisy traffic
- **Feature importance** gives analysts interpretable IoC signals
- Handles high-dimensional, correlated network features natively

### Key Security Metrics
- **Recall > 99.99%** → fewer than 1 in 10,000 DDoS flows missed (false negatives)
- **Precision > 99.99%** → minimal alert fatigue for security analysts
- **False negatives** (missed attacks) are far more dangerous than false positives in this domain — the model is tuned to minimise them

### Production Limitations
- CIC-IDS2017 is a controlled lab dataset — real-world traffic is noisier
- Zero-day DDoS variants unseen during training may evade detection
- Requires CICFlowMeter for live PCAP → feature extraction pipeline
- Periodic retraining is required as attack patterns evolve

---

## 🔮 Future Work

- [ ] LSTM/Transformer for temporal traffic sequence modelling
- [ ] Online learning for real-time model updates
- [ ] Multi-class detection (Brute Force, Web Attack, Infiltration, Port Scan)
- [ ] CICFlowMeter live PCAP integration
- [ ] Docker containerisation
- [ ] SIEM/Splunk webhook integration
- [ ] Adversarial traffic robustness testing

---

## 📚 References

1. Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
2. [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity
3. [scikit-learn Documentation](https://scikit-learn.org)
4. [Streamlit Documentation](https://docs.streamlit.io)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made for CLO4 — Information Security**

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/CLO4-IDS-ML-Solution?style=social)](https://github.com/YOUR_USERNAME/CLO4-IDS-ML-Solution)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/CLO4-IDS-ML-Solution?style=social)](https://github.com/YOUR_USERNAME/CLO4-IDS-ML-Solution/fork)

*If this project helped you, consider giving it a ⭐*

</div>
