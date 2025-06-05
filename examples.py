from utils.data import prepare_cifar10_loaders
    

def main():
    train_loader, valid_loader, test_loader = prepare_cifar10_loaders(batch_size=50,
                                                                    data_path="Data/CIFAR10",
                                                                    shuffle=True
                                                                    )
    
    
    print(len(train_loader.dataset))
    print(len(valid_loader.dataset))
    print(len(test_loader.dataset))

if __name__ == "__main__":
    main()
