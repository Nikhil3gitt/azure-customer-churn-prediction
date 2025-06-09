"""Check model drift and trigger Azure ML pipeline retraining when necessary."""

import os, requests, datetime, json

def check_and_trigger(endpoint_url, threshold=0.02):
    # (Placeholder logic) Fetch weekly AUC from monitoring service
    resp = requests.get(f"{endpoint_url}/metrics")
    auc = json.loads(resp.text)['auc']
    baseline = float(os.getenv("BASELINE_AUC", "0.88"))
    if baseline - auc > threshold:
        print("AUC dropped, triggering retraining...")
        # Call Azure ML pipeline REST endpoint
        retrain_resp = requests.post(os.getenv("RETRAIN_PIPELINE_REST"))
        print(retrain_resp.status_code, retrain_resp.text)
    else:
        print("Model within expected performance.")

if __name__ == "__main__":
    check_and_trigger(os.environ.get("ENDPOINT_URL"))
