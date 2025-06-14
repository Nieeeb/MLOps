import wandb


def wandb_init(params):
    run = wandb.init(
        entity="deep-learning-mini-project",
        project="MLOps",
        config=params,
        group=params.get("run_name"),
        id=params.get("run_name"),
        resume="allow",
    )
    return run


def wandb_log(epoch, loss_avg, val_loss):
    wandb.log(
        {
            "Epoch": epoch,
            "Training loss average": loss_avg,
            "Validation loss": val_loss,
        }
    )
