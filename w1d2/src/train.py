import time
import einops
import torch as t
from tqdm import tqdm 
import wandb

def train(model, optimizer, trainloader, testloader, criterion, num_epochs=10, device = "mps"):

    examples_seen = 0
    since = time.time()

    print("Beginning Training")
    data = next(iter(testloader))
    example_output = model(data[0])[:3]
    print("expected results")
    print(data[1][:3])
    print("current, bad, output")
    print(t.argmax(example_output, dim = -1))
    print("="*30)
    
    model.to(device)
    best_model = None
    best_acc = 0.0

    wandb.watch(model, criterion=criterion, log="all", log_freq=10, log_graph=True)

    for epoch in range(num_epochs):
        
        
        model.train()
        
        running_loss = 0.0
        progress_bar = tqdm(trainloader)

        for (x, y)  in progress_bar:
            
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            training_loss = criterion(einops.rearrange(logits, "b s v -> (b s) v"), y.flatten())
            
            progress_bar.set_description(desc="Epoch {} Training Loss {}".format(epoch, training_loss))

            examples_seen += len(x)
            wandb.log({"train_loss": training_loss, "elapsed": time.time() - since}, step=examples_seen)
            
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            running_loss += training_loss.item() * x.size(0) # scale to n in batch

        epoch_loss = running_loss / len(trainloader.dataset)
        test_accuracy = test_model(model, testloader, device)
        wandb.log({"test accuracy": test_accuracy})

        print('Epoch {} Loss: {:.4f} Test Accuracy {:.4f}'.format(epoch, epoch_loss, test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model = model

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))   

    filename = f"best_decoder_only_rev_sequence_model.pt"
    print(f"Saving best model to {filename}")
    t.save(best_model.state_dict(), filename)

    return model 


def test_model(model, testloader, device):

    num_correct = 0 
    num_total = 0

    with t.inference_mode():
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
        