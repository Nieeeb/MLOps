from coco import COCODataset, collateFunction
import hydra
from torch.utils.data import DataLoader
from utils import load_datasets


@hydra.main(config_path="../config", config_name="config")
def main(args):

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    
    #train_dataloader = DataLoader(train_dataset, 
    #    batch_size=args.batchSize, 
    #    shuffle=True, 
    #    collate_fn=collateFunction, 
    #    num_workers=args.numWorkers)
    
    val_dataloader = DataLoader(val_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        num_workers=args.numWorkers)

    test_dataloader = DataLoader(test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collateFunction,
        num_workers=args.numWorkers)
    
    


    epochs = 1
    for epoch in range(epochs):
        for batch, (images, targets) in val_dataloader:
            print(targets)
        
if __name__ == '__main__':
    main()