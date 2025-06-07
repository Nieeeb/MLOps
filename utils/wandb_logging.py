import wandb


def wandb_init(params):
    wandb.init(
        entity="deep-learning-mini-project",
        project="MLOps",
        config=params,
        group=params.get("run_name"),
        id=params.get("run_name"),
        resume="allow",
    )


def wandb_log(epoch, loss_avg, val_loss):
    wandb.log(
        {
            "Epoch": epoch,
            "Training loss average": loss_avg,
            "Validation loss": val_loss,
        }
    )


if __name__ == "__main__":
    params = {"run_name": "no_train_test"}
    wandb_init(params)
    for i in range(1000):
        wandb_log(epoch=i, val_loss=i + 1, loss_avg=i + 2)
