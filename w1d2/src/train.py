import time
import einops
import torch as t

def train(model, optimizer, trainloader, testloader, criterion, num_epochs=10, device = "mps"):

    since = time.time()

    print("Beginning Training")
    data = next(iter(testloader))
    example_output = model(data[0])[:3]
    print(data[1][:3])
    print(t.argmax(example_output, dim = -1))
    print("="*30)

    for epoch in range(num_epochs):
        
        model.to(device)
        model.train()
        
        running_loss = 0.0

        for batch, (x, y)  in enumerate(trainloader):
            
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            #print(preds.shape)
            preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")
            #print(preds_rearranged.shape)
            y_rearranged = einops.rearrange(y, "b s -> (b s)")
            #print(y_rearranged.shape)

            training_loss = criterion(preds_rearranged, y_rearranged)

            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            running_loss += training_loss.item() * x.size(0) # scale to n in batch

        epoch_loss = running_loss / len(trainloader.dataset)
        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))

        data = next(iter(testloader))
        example_output = model(data[0])[:3]
        print(data[1][:3])
        print(t.argmax(example_output, dim = -1))
        print("="*30)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))   

    return model 
