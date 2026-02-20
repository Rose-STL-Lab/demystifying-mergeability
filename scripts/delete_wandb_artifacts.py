import wandb

api = wandb.Api()

# Replace <entity> and <project> with your details
entity = "lzhou00-sapienza-universit-di-roma"
project = "model_merging"

runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(f"Processing run: {run.name}")
    for artifact in run.logged_artifacts():
        print(f"  Deleting artifact: {artifact.name}:{artifact.version}")
        artifact.delete(delete_aliases=True)