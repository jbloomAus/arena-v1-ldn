import time
import einops
import torch as t
from tqdm import tqdm 


def train(model, optimizer, trainloader, testloader, criterion, num_epochs=10, device = "mps"):

    since = time.time()

    print("Beginning Training")
    data = next(iter(testloader))
    example_output = model(data[0])[:3]
    print("expected results")
    print(data[1][:3])
    print("current, bad, output")
    print(t.argmax(example_output, dim = -1))
    print("="*30)
    

    for epoch in range(num_epochs):
        
        model.to(device)
        model.train()
        
        running_loss = 0.0
        training_loss = 0.0
        progress_bar = tqdm(trainloader)

        for (x, y)  in progress_bar:
            
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")
            y_rearranged = einops.rearrange(y, "b s -> (b s)")
            training_loss = criterion(preds_rearranged, y_rearranged)
            
            progress_bar.set_description(desc="Epoch {} Training Loss {}".format(epoch, training_loss))
            
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            running_loss += training_loss.item() * x.size(0) # scale to n in batch

        epoch_loss = running_loss / len(trainloader.dataset)
        test_accuracy = test_model(model, testloader, device)
        print('Epoch {} Loss: {:.4f} Test Accuracy {:.4f}'.format(epoch, epoch_loss, test_accuracy))

        # data = next(iter(testloader))
        # example_output = model(data[0].to(device))[:3]
        # print(data[1][:3])
        # print(t.argmax(example_output, dim = -1))
        # print("="*30)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))   

    return model 


def test_model(model, testloader, device):

    num_correct = 0 
    num_total = 0

    for (x,y) in testloader:

        x.to(device)
        y.to(device)

        preds = model(x)
        
        preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")
        y_rearranged = einops.rearrange(y, "b s -> (b s)")
        
        digit_predictions = (t.argmax(preds_rearranged, dim = -1))
        num_correct += (digit_predictions == y_rearranged).sum().item()
        num_total += y_rearranged.size(0)

    accuracy = 100*num_correct / num_total
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy
        