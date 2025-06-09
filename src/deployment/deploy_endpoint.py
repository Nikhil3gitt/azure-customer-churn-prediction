"""Register the trained model and deploy to an Azure ML managed online endpoint."""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
import os, joblib

def deploy(workspace_name, resource_group, subscription_id):
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )

    model = ml_client.models.create_or_update(
        Model(
            name="lightgbm-customer-churn",
            path="models/lightgbm_churn.pkl",
            type="mlflow_model",
        )
    )

    endpoint = ManagedOnlineEndpoint(
        name="churn-endpoint",
        auth_mode="key"
    )
    ml_client.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint.name,
        model=model.id,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    ml_client.begin_create_or_update(deployment).result()
    ml_client.online_endpoints.begin_traffic_shift(endpoint.name, {"blue": 100}).result()

if __name__ == "__main__":
    deploy(
        workspace_name=os.environ.get("AML_WORKSPACE"),
        resource_group=os.environ.get("AML_RG"),
        subscription_id=os.environ.get("AZ_SUBSCRIPTION_ID"),
    )
