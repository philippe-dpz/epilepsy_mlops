import os
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Default"
MODEL_NAME = "epilepsy_model"
METRIC_NAME = "val_accuracy"
PRODUCTION_DIR = "production"

def get_best_run(experiment_id, client):
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )
    return runs[0] if runs else None

def get_production_model_metric(model_name, client):
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None, None
    version = versions[0]
    run = client.get_run(version.run_id)
    metric = run.data.metrics.get(METRIC_NAME)
    return metric, version

def register_model(run, model_name):
    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"✅ Registered new model version: {result.version}")
    return result  # Return full ModelVersion object

def promote_model_if_better(new_metric, new_version_number, current_metric, client):
    if current_metric is None or new_metric > current_metric:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version_number,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"🚀 Promoted model version {new_version_number} to Production (val_accuracy: {new_metric})")
        return True
    else:
        print(f"⚠️ New model (val_accuracy: {new_metric}) is not better than current Production (val_accuracy: {current_metric}) — not promoted.")
        return False

def save_production_model_to_dvc(model_name, version_obj, dest_path=PRODUCTION_DIR, client=None):
    if os.path.exists(dest_path):
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    else:
        os.makedirs(dest_path, exist_ok=True)

    print(f"⬇️ Downloading model version {version_obj.version} to '{dest_path}'")
    # ✅ Use the correct artifact path
    client.download_artifacts(run_id=version_obj.run_id, path="epilepsy_model", dst_path=dest_path)
    print(f"✅ Model saved to '{dest_path}'")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Update if using remote tracking server
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
        exit(1)

    best_run = get_best_run(experiment.experiment_id, client)
    if not best_run:
        print("❌ No successful runs found.")
        exit(1)

    best_metric = best_run.data.metrics.get(METRIC_NAME)
    print(f"🔍 Best run: {best_run.info.run_id} with {METRIC_NAME}: {best_metric}")

    current_metric, current_version = get_production_model_metric(MODEL_NAME, client)
    if current_metric:
        print(f"📦 Current production version: {current_version.version} with {METRIC_NAME}: {current_metric}")
    else:
        print("ℹ️ No production model yet.")

    new_model_version = register_model(best_run, MODEL_NAME)

    promote_model_if_better(best_metric, new_model_version.version, current_metric, client)

    # Always download best model for DVC tracking
    save_production_model_to_dvc(MODEL_NAME, new_model_version, PRODUCTION_DIR, client)
