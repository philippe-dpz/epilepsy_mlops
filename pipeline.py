from prefect import flow, task
import subprocess

@task
def pull_raw_data():
    # Pull from the remote DVC cache (DagsHub)
    subprocess.run(["dvc", "pull", "data/raw.dvc"], check=True)
    print("✅ Pulled raw data from DagsHub")

@task
def run_preprocessing():
    import subprocess
    subprocess.run(["docker", "compose", "run", "--rm", "preprocessing"], check=True)

@task
def sync_dvc_updates():
    import subprocess
    subprocess.run(["dvc", "add", "data/processed", "data/patients"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Update processed data and patients"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    
@flow
def epilepsy_pipeline():
    pull_raw_data()
    run_preprocessing()
    sync_dvc_updates()

if __name__ == "__main__":
    epilepsy_pipeline()