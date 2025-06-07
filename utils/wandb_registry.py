from pathlib import Path
import wandb


def wandb_model_reg(run_dir):
    run = wandb.init(project="collection-linking-quickstart")

    artifact_filepath = Path(run_dir) / "checkpoint_last.pt"
    if not artifact_filepath.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {artifact_filepath}"
        )

    artifact = wandb.Artifact(
        name="model_checkpoint_last",
        type="model",
        description="Last checkpoint model",
    )
    artifact.add_file(str(artifact_filepath))

    run.log_artifact(artifact)

    run.link_artifact(artifact, target_path="wandb-registry-model/Models")

    run.finish()
