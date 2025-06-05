import wandb


def wandb_init(params):
    wandb.init(
        entity="deep-learning-mini-project",
        project="MLOps",
        config=params,
        group=params.get("run_name"),
        id=params.get("run_name"),
    )


def wandb_log(epoch, val_loss, loss_avg, loss):
    wandb.log(
        {
            "Training step": epoch,
            "Training loss average": loss_avg,
            "Validatoin loss": val_loss,
        }
    )
