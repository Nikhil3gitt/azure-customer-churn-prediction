"""Train LightGBM model locally or within Azure ML."""

import argparse, yaml, lightgbm as lgb, pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_parquet(cfg['dataset_path'])
    X = df.drop(columns=[cfg['target_column']])
    y = df[cfg['target_column']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg['test_size'], random_state=cfg['random_state'], stratify=y
    )
    model = lgb.LGBMClassifier(**cfg['model']['params'])
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    print(f"ROC‑AUC: {auc:.4f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lightgbm_churn.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_train.yml")
    args = parser.parse_args()
    main(args.config)
