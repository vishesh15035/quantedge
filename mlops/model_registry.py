import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

class ModelRegistry:
    """
    Promotes models through: None → Staging → Production
    Keeps track of champion model per strategy type
    """
    STAGES = ["None", "Staging", "Production", "Archived"]

    def __init__(self, tracking_uri="./mlflow-artifacts"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str,
                       description: str = "") -> str:
        uri    = f"runs:/{run_id}/model"
        result = mlflow.register_model(uri, model_name)
        self.client.update_registered_model(
            name=model_name, description=description)
        print(f"[Registry] Registered: {model_name} v{result.version}")
        return result.version

    def promote_to_staging(self, model_name: str, version: str):
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Staging")
        print(f"[Registry] {model_name} v{version} → Staging")

    def promote_to_production(self, model_name: str, version: str):
        # Archive current production first
        try:
            prod = self.client.get_latest_versions(model_name, stages=["Production"])
            for p in prod:
                self.client.transition_model_version_stage(
                    name=model_name, version=p.version, stage="Archived")
                print(f"[Registry] Archived old production: v{p.version}")
        except Exception:
            pass
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Production")
        print(f"[Registry] {model_name} v{version} → Production ✓")

    def get_production_model(self, model_name: str):
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        except Exception as e:
            print(f"[Registry] No production model: {e}")
        return None

    def list_models(self) -> pd.DataFrame:
        try:
            models = self.client.search_registered_models()
            rows   = []
            for m in models:
                for v in m.latest_versions:
                    rows.append({
                        "name":    m.name,
                        "version": v.version,
                        "stage":   v.current_stage,
                        "created": datetime.fromtimestamp(
                            v.creation_timestamp/1000).strftime("%Y-%m-%d %H:%M"),
                    })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
