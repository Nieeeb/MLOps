import yaml
import argparse
import os
import warnings
import torch
from utils import util
import wandb
from utils.model_tools import load_model
import torch.multiprocessing as mp
from datetime import timedelta
from utils.data import prepare_cifar10_loaders
import torchvision
import torch.nn as nn
from utils.wandb_logging import wandb_init, wandb_log


warnings.filterwarnings("ignore")


def main():
    # Loading args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", default="configs/params.yaml", type=str)
    parser.add_argument("--world_size", default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(f"Local rank: {args.local_rank}")
    print(f"World size: {args.world_size}")

    # Setting random seed for reproducability
    # Seed is 0
    util.setup_seed()

    # Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load(cf_file.read())

    # Creating training instances for each GPU
    mp.spawn(train, args=(args, params), nprocs=args.world_size, join=True)


# Function for defining machine and port to use for DDP
# Sets up the process group
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "61111"
    # os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', 0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=1)
    )


# Method to gracefully destory the process group
# If this is not done, problems may arise on future runs
def cleanup():
    torch.distributed.destroy_process_group()


# Function for training an entire epoch
# Logs to wandb
def train_epoch(
    args,
    params,
    model,
    optimizer,
    train_loader,
    train_sampler,
    criterion,
    epoch,
):
    m_loss = util.AverageMeter()
    # If in DDP, sampler needs current epoch
    # Used to determine which data shuffle to use if GPUs get desynced
    if args.world_size > 1:
        train_sampler.set_epoch(epoch)

    model.train()

    for batchidx, (samples, targets) in enumerate(train_loader):
        samples, targets = samples.to(args.local_rank), targets.to(
            args.local_rank
        )
        if args.local_rank == 0:
            print(f"Size of samples in training: {samples.shape}")

        optimizer.zero_grad()

        samples = (
            samples.float() / 255
        )  # Input images are 8 bit single channel images. Converts to 0-1 floats

        outputs = model(samples)  # forward pass
        loss = criterion(outputs, targets)

        m_loss.update(loss.item(), samples.size(0))

        loss *= params.get("batch_size")  # loss scaled by batch_size
        loss *= (
            args.world_size
        )  # gradient averaged between devices in DDP mode

        loss.backward()  # Backpropagation

        del loss  # Deletes loss to save memory

        optimizer.step()  # Steps the optimizer

    # scheduler.step()  # Step learning rate scheduler

    return m_loss


# Function that validates the model on a validation set
# Intended to be used during training and not for testing performance of model
def validate_epoch(
    args,
    params,
    model,
    validation_loader,
    validation_sampler,
    criterion,
    epoch,
    resize=False,
):
    print(
        f"Beginning epoch validation for epoch {epoch + 1} on GPU {args.local_rank}"
    )
    v_loss = util.AverageMeter()

    print("Defines v_loss")
    # If in DDP, sampler needs current epoch
    # Used to determine which data shuffle to use if GPUs get desynced
    if args.world_size > 1:
        validation_sampler.set_epoch(epoch)

    print("validation sampler set up")
    # Iterates through validation set
    # Disables gradient calculations
    print(f"Samples in validation loader: {len(validation_loader)}")

    model.eval()
    # with torch.no_grad():
    #     print("Can do torch no grad")
    for batchidx, (samples, targets) in enumerate(validation_loader):
        print("reached the loop where we enumerate the validation loader")
        # Sending data to appropriate GPU
        samples, targets = samples.to(args.local_rank), targets.to(
            args.local_rank
        )

        if args.local_rank == 0:
            print(f"Val batch nr: {batchidx}, size: {samples.shape}")

        # if resize:
        #     resize = torchvision.transforms.Resize((128, 128))
        #     samples = resize(samples)

        samples = (
            samples.float() / 255
        )  # Input images are 8 bit single channel images. Converts to 0-1 floats

        outputs = model(samples)  # Forward pass

        vloss = criterion(outputs, targets)

        torch.distributed.reduce(
            vloss, torch.distributed.ReduceOp.AVG
        )  # Syncs loss and takes the average across GPUs
        v_loss.update(vloss.item(), samples.size(0))

        if args.local_rank == 0:
            print(f"can finish validation iteration nr: {batchidx}")

        del outputs
        del vloss

    print(f"GPU {args.local_rank} has completed validation")

    return v_loss


def train(rank, args, params):
    try:
        # Defining world size and creating/connecting to DPP instance
        args.local_rank = rank
        setup(rank, args.world_size)

        # Loading model
        # Loads if a valid checkpoint is found, otherwise creates a new model
        # model, optimizer, scheduler, starting_epoch = load_or_create_state(args, params)

        net = load_model(train=True)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        # if starting_epoch + 1 >= params.get('epochs'):
        #     print(f"Already trained for {params.get('epochs')} epochs. Exiting")
        #     exit

        (
            train_loader,
            valid_loader,
            test_loader,
            train_sampler,
            valid_sampler,
            test_sampler,
        ) = prepare_cifar10_loaders()

        criterion = nn.CrossEntropyLoss()

        # Init Wandb
        if args.local_rank == 0:
            wandb.init()

        if args.local_rank == 0:
            wandb.log({"config": args.args_file})

        # Pauses all worker threads to sync up GPUs before training
        torch.distributed.barrier()

        # Begin training
        if args.local_rank == 0:
            print("Beginning training...")
        for epoch in range(params.get("epochs")):
            if args.local_rank == 0:
                print(f"Traning for epoch {epoch + 1}")
            m_loss = train_epoch(
                args,
                params,
                model=net,
                optimizer=optimizer,
                train_loader=train_loader,
                train_sampler=train_sampler,
                criterion=criterion,
                epoch=epoch,
            )
            if args.local_rank == 0:
                print(f"Validation for epoch {epoch + 1}")
            v_loss = validate_epoch(
                args,
                params,
                model=net,
                validation_loader=valid_loader,
                validation_sampler=valid_sampler,
                criterion=criterion,
                epoch=epoch,
            )

            run_dir = params.get("run_dir")

            ckpt = {
                "epoch": epoch,
                "model": net.state_dict(),
                "optim": optimizer.state_dict(),
                "train_loss": m_loss,
                "val_loss": v_loss,
            }
            torch.save(
                ckpt, os.path.join(run_dir, f"checkpoint_{epoch:03d}.pt")
            )
            torch.save(ckpt, os.path.join(run_dir, "checkpoint_last.pt"))

            if args.local_rank == 0:
                print(
                    f"Validation for epoch {epoch} complete. Val Loss is at: {v_loss.avg}"
                )
                wandb_log(epoch=epoch, loss_avg=m_loss, val_loss=v_loss)

                del m_loss
                del v_loss

        # Training complete
        if args.local_rank == 0:
            print(
                f"Training Completed succesfully\nTrained {params.get('epochs')} epochs"
            )

        torch.distributed.barrier()  # Pauses all worker threads to sync up GPUs
        cleanup()  # Destroy DDP process group

    except Exception as e:
        cleanup()
        print(e)
        exit


if __name__ == "__main__":
    main()
