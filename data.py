
import torch
from torchvision import datasets, transforms

def getDataLoader(train, args):

    ### Define data transform
    inputTrsform = transforms.Compose([
        transforms.Resize((520, 480)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    outputTrsform = transforms.Compose([
        transforms.Resize((520, 480)),
        transforms.ToTensor()
    ])

    ### Training dataset
    if train == True :
        trainDataset = datasets.VOCSegmentation(
            "./data/train",
            image_set="train",
            download=False,
            transform=inputTrsform,
            target_transform=outputTrsform
        )

        return torch.utils.data.DataLoader(
            dataset=trainDataset,
            batch_size = args.batchSize,
            shuffle=True,
            num_workers=args.numWorkers,
            drop_last=True
        )

    ### Evaluating dataset
    else :
        testDataset = datasets.VOCSegmentation(
            "./data/eval",
            image_set="val",
            download=False,
            transform=inputTrsform,
            target_transform=outputTrsform
        )

        return torch.utils.data.DataLoader(
            dataset=testDataset,
            batch_size = args.batchSize,
            shuffle=False,
            num_workers=args.numWorkers,
            drop_last=False
        )
