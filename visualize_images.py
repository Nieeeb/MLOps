from utils.data import prepare_cifar10_loaders
from utils.util import visualize_imgs_from_loader
import multiprocessing


def main():
    train_loader, *_ = prepare_cifar10_loaders(
        shuffle=False, drift_test=True, drift_index=0
    )  #
    visualize_imgs_from_loader(train_loader, batch_size=50)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
