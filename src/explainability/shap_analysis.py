"""Generate SHAP explanations and write to Cosmos DB."""

import lightgbm as lgb, joblib, shap, pandas as pd, argparse, json
from azure.cosmos import CosmosClient

def explain(model_path, sample_path, cosmos_conn, db_name, container):
    model = joblib.load(model_path)
    X = pd.read_parquet(sample_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    client = CosmosClient.from_connection_string(cosmos_conn)
    cont = client.get_database_client(db_name).get_container_client(container)
    for i, row in X.iterrows():
        cont.upsert_item({
            "id": str(i),
            "customer_id": row['customer_id'],
            "shap": shap_values[1][i].tolist()
        })
    print("SHAP values stored in Cosmos DB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lightgbm_churn.pkl")
    parser.add_argument("--sample", default="data/processed/telco_churn_sample.parquet")
    args = parser.parse_args()
    explain(args.model, args.sample,
            os.environ["COSMOS_CONN"], "churn", "shap_values")
