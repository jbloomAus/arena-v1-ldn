import multiprocessing
import os
import time
import wandb

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import resnet34
from tqdm import tqdm

from d2 import Linear
from d3 import ResNet34, copy_weights

data_dir = "./w0d3/hymenoptera_data/"

learning_rate = 0.001
epochs = 1
batch_size = 8
num_classes = 2
num_workers = multiprocessing.cpu_count() - 1
device = "cpu"

wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size,
  "num_workers": num_workers,
}

def train_model(model,
                optimizer,
                trainloader,
                testloader,
                criterion,
                num_epochs=10,
                device="mps"):

    since = time.time()
    model.to(device)
    for epoch in tqdm(range(num_epochs)):

        model.train()
        running_loss = 0.0
        running_corrects = 0

        # training
        for (x, y) in tqdm(trainloader):

            x = x.to(device)
            y = y.to(device)
            training_loss = 0

            preds = model(x)
            training_loss = criterion(preds, y)
            train_corrects = torch.sum(preds.argmax(dim=1) == y.data)

            wandb.log({"train_loss": training_loss})
            wandb.log({"train_acc": train_corrects / y.shape[0] })

            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += training_loss.item() * x.size(0)  # scale to n in batch
            running_corrects += train_corrects

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)

        print('{}: training: Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))


        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # testing
        for (x, y) in tqdm(testloader):

            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            test_loss = criterion(preds, y)
            test_corrects = torch.sum(preds.argmax(dim=1) == y.data)

            running_loss += test_loss.item() * x.size(0)  # scale to n in batch
            running_corrects += test_corrects

            wandb.log({"test_loss": test_loss})
            wandb.log({"test_acc": test_corrects / y.shape[0]})

        epoch_loss = running_loss / len(testloader.dataset)
        epoch_acc = running_corrects.double() / len(testloader.dataset)

        print('{}: test: Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    return model

if __name__ == '__main__':


    wandb.init(
        project="Antman-and-the-Wasp--Finetunemania - Fixed",
        name = "Pretty Graphs",
        entity="arena-ldn"
        )


    print("Initializing model and copy weights...")
    myresnet = ResNet34()
    myresnet = copy_weights(myresnet, pretrained_resnet=resnet34(weights="DEFAULT"))
    myresnet.linear = Linear(512, num_classes)
    input_size = 244

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Initializing Datasets and Dataloaders...")
    traindata = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=transform_train)
    trainloader = torch.utils.data.DataLoader(traindata,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    testdata = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                    transform=transform_val)
    testloader = torch.utils.data.DataLoader(testdata,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)

    # let's get the optimizer working

    # Send the model to GPU
    myresnet = myresnet.to(device)

    # Don't update anything except last layer
    for name, param in myresnet.named_parameters():
        if name.startswith("linear"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("Params to learn:")
    params_to_update = []
    for name, param in myresnet.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    myresnet = train_model(myresnet,
                        optimizer_ft,
                        trainloader,
                        testloader,
                        criterion,
                        num_epochs=epochs,
                        device=device)
