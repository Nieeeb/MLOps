from pathlib import Path
import wandb


def wandb_model_reg(run_dir, run):
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

    logged_artifact = run.log_artifact(artifact)

    run.link_artifact(
        logged_artifact, target_path="wandb-registry-model/Models"
    )

    wandb.finish()
