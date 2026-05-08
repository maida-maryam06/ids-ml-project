"""
model_trainer.py — Standalone Model Training Script
====================================================
Run this instead of Jupyter Notebook if you prefer the terminal.

Usage:
    python backend/model_trainer.py --data data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Output:
    backend/model.pkl
    backend/scaler.pkl
    backend/label_encoder.pkl
    backend/feature_names.pkl
    frontend/  (all evaluation charts)
"""

import argparse
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
os.makedirs(FRONTEND_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    print(f'\n[1/5] Loading dataset: {path}')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()          # CIC-IDS2017 has leading spaces
    print(f'      Shape: {df.shape}')
    print(f'      Label counts:\n{df["Label"].value_counts().to_string()}')
    return df


def preprocess(df: pd.DataFrame):
    print('\n[2/5] Preprocessing...')

    # Replace inf with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f'      Dropped {before - len(df)} rows with NaN/Inf. Remaining: {len(df):,}')

    # Encode labels
    le = LabelEncoder()
    df['Label_encoded'] = le.fit_transform(df['Label'])
    print(f'      Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    # Features & target
    X = df.drop(columns=['Label', 'Label_encoded']).select_dtypes(include=[np.number])
    y = df['Label_encoded']

    print(f'      Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}')
    return X, y, le


def split_and_scale(X, y):
    print('\n[3/5] Splitting and scaling...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f'      Train: {X_train_s.shape}  |  Test: {X_test_s.shape}')
    return X_train_s, X_test_s, y_train, y_test, scaler


def train_models(X_train, y_train):
    print('\n[4/5] Training models...')

    print('      → Random Forest (n=100, max_depth=20)...')
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print('      ✓ Random Forest done')

    print('      → Decision Tree (max_depth=10)...')
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    print('      ✓ Decision Tree done')

    return rf, dt


def evaluate(rf, dt, X_test, y_test, le):
    print('\n[5/5] Evaluating...')
    results = {}

    for model, name in [(rf, 'Random Forest'), (dt, 'Decision Tree')]:
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        f1   = f1_score(y_test, y_pred, average='weighted')

        print(f'\n  ── {name} ──────────────────')
        print(f'     Accuracy:  {acc*100:.2f}%')
        print(f'     Precision: {prec*100:.2f}%')
        print(f'     Recall:    {rec*100:.2f}%')
        print(f'     F1-Score:  {f1*100:.2f}%')
        print(f'\n{classification_report(y_test, y_pred, target_names=le.classes_)}')

        results[name] = {
            'preds': y_pred,
            'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        }

    return results


def save_charts(df, X, rf, dt, y_test, results, le):
    print('\n  Saving charts to frontend/...')

    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['Label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution (Count)', fontweight='bold')
    for i, (lbl, val) in enumerate(counts.items()):
        axes[0].text(i, val + 500, f'{val:,}', ha='center', fontweight='bold')
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=[0.05]*len(counts))
    axes[1].set_title('Class Distribution (%)', fontweight='bold')
    plt.suptitle('CIC-IDS2017 — DDoS vs BENIGN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('     ✓ class_distribution.png')

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, color) in zip(axes, [('Random Forest', 'Blues'), ('Decision Tree', 'Greens')]):
        cm = confusion_matrix(y_test, results[name]['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    linewidths=1, linecolor='white')
        ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Model Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('     ✓ confusion_matrices.png')

    # 3. Model comparison
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    rf_vals = [results['Random Forest']['metrics'][m] for m in metrics_names]
    dt_vals = [results['Decision Tree']['metrics'][m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, rf_vals, w, label='Random Forest', color='#3498db', edgecolor='black')
    b2 = ax.bar(x + w/2, dt_vals, w, label='Decision Tree', color='#2ecc71', edgecolor='black')
    ax.set_ylim(0.85, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_names])
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('     ✓ model_comparison.png')

    # 4. Feature importance
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues_r')
    plt.title('Top 15 Features — Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FRONTEND_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('     ✓ feature_importance.png')

    # 5. Feature distributions
    key_feats = [f for f in
                 ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s', 'Flow Packets/s']
                 if f in X.columns]
    if key_feats:
        fig, axes = plt.subplots(1, len(key_feats), figsize=(16, 4))
        if len(key_feats) == 1:
            axes = [axes]
        for ax, feat in zip(axes, key_feats):
            for label, color in [('BENIGN', '#2ecc71'), ('DDoS', '#e74c3c')]:
                mask = df['Label'] == label
                vals = df.loc[mask, feat].replace([np.inf, -np.inf], np.nan).dropna()
                ax.hist(vals, bins=50, alpha=0.6, color=color, label=label, density=True)
            ax.set_title(feat, fontsize=9)
            ax.legend(fontsize=8)
        plt.suptitle('Feature Distributions: BENIGN vs DDoS', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FRONTEND_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('     ✓ feature_distributions.png')


def save_artifacts(rf, scaler, le, X):
    print('\n  Saving model artifacts to backend/...')
    joblib.dump(rf,            os.path.join(BASE_DIR, 'model.pkl'))
    joblib.dump(scaler,        os.path.join(BASE_DIR, 'scaler.pkl'))
    joblib.dump(le,            os.path.join(BASE_DIR, 'label_encoder.pkl'))
    joblib.dump(list(X.columns), os.path.join(BASE_DIR, 'feature_names.pkl'))
    print('     ✓ model.pkl')
    print('     ✓ scaler.pkl')
    print('     ✓ label_encoder.pkl')
    print('     ✓ feature_names.pkl')


def main():
    parser = argparse.ArgumentParser(description='Train the IDS ML model.')
    parser.add_argument(
        '--data',
        default=os.path.join(PROJECT_ROOT, 'data',
                             'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'),
        help='Path to the CIC-IDS2017 CSV file'
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f'\n❌ Dataset not found: {args.data}')
        print('   Place the CSV in the data/ folder and try again.')
        sys.exit(1)

    print('=' * 60)
    print('  SecureNet IDS — Model Trainer')
    print('=' * 60)

    df           = load_data(args.data)
    X, y, le    = preprocess(df)
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, y)
    rf, dt       = train_models(X_tr, y_tr)
    results      = evaluate(rf, dt, X_te, y_te, le)

    save_charts(df, X, rf, dt, y_te, results, le)
    save_artifacts(rf, scaler, le, X)

    print('\n' + '=' * 60)
    print('  ✅ Training complete!')
    print(f'     Random Forest Accuracy: {results["Random Forest"]["metrics"]["accuracy"]*100:.2f}%')
    print('     Run: python backend/app.py  →  open frontend/index.html')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
