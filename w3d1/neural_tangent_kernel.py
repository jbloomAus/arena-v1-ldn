
# %% 
import torch as t 
import torch.nn as nn 
import pandas as pd 
import plotly.express as px 
from torch.optim import SGD 
import einops

class NeuralNetwork(nn.Module):
    ''' A simple neural network with 2 hidden layers, Relu activation '''

    def __init__(self, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(1, hidden_size)
        t.nn.init.normal_(self.linear1.weight, std=1.0)
        self.linear2 = nn.Linear(hidden_size, 1)
        t.nn.init.normal_(self.linear2.weight, std=1/hidden_size**0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x

def map_current_kernel(neural_networks, x = t.linspace(-4, 4, 100)):
    y = [] 
    for neural_network in neural_networks:
        y_tmp = []
        for i in x:
            y_tmp.append(neural_network(i.unsqueeze(0)))
        y.append(y_tmp)
    y = t.tensor(y)
    return y 

def loss(x, y):
    return (x - y)**2

norm2 = lambda x: (x**2).sum()

def relative_change_in_model_jacobian(model, original_model_y, x):
    jacobian = t.autograd.functional.jacobian(model, x[0].unsqueeze(0))
    hessian = t.autograd.functional.hessian(model, x[0].unsqueeze(0))
    dist = norm2(model(x[0].unsqueeze(0))-original_model_y)
    return dist*hessian/norm2(jacobian)

def training_loop(neural_networks, original_model_preds, optimizers, x, y, steps = 100):
    '''
    Write a training loop which updates each nn after prediction on x
    Inputs:
        neural_networks: list of neural networks
        optimizers: list of optimizers
        x: input data
        y: target data
        steps: number of training steps
    Output:
        all_ys: tensor of shape (steps, len(neural_networks), len(x))
        weights: tensor of shape (steps, len(neural_networks), hidden_size)
    '''
    
    all_ys = []
    weights = []
    relative_changes_in_jacobians = []
    for step in range(steps):
        weights_step = []
        jacobian_changes_step = []
        for i in range(len(neural_networks)):
            y_pred = neural_networks[i](x.unsqueeze(0).T)
            l = loss(y_pred, y.unsqueeze(0).T).mean()
            l.backward()
            optimizers[i].step()
            neural_networks[i].zero_grad()
            weights_step.append(neural_networks[i].linear2.weight.clone())
            jacobian_changes_step.append(
                relative_change_in_model_jacobian(neural_networks[i], original_model_preds[i], x)
            )

        with t.inference_mode():
            all_ys.append(map_current_kernel(neural_networks, x = t.linspace(-4, 4, 100)))  
        weights.append(t.stack(weights_step))
        relative_changes_in_jacobians.append(t.stack(jacobian_changes_step))


    all_ys = t.stack(all_ys)
    weights = t.stack(weights).squeeze(2)
    relative_changes_in_jacobians = t.stack(relative_changes_in_jacobians)
    return all_ys, weights, relative_changes_in_jacobians

def plot_weights(weights):
    fig  = px.imshow(weights.detach(), template="plotly_dark", animation_frame=0)
    fig.show()

def plot_model_functions(all_ys, x, y):
    n_networks = all_ys.shape[1]
    steps = all_ys.shape[0]
    x_range = t.linspace(-4, 4, 100)
    flattened_ys = all_ys.flatten()
    flat_df = pd.DataFrame(flattened_ys)
    flat_df.columns = ['y']
    flat_df['x'] = [x for step in range(steps) for net in range(n_networks) for x in x_range]
    flat_df['network'] = [net for step in range(steps) for net in range(n_networks) for x in x_range]
    flat_df['step'] = [step for step in range(steps) for net in range(n_networks) for x in x_range]

    fig = px.line(flat_df, x = 'x', y = 'y', color = 'network', animation_frame = 'step')
    # add original x and y as scatter
    fig.add_scatter(x = x, y = y, mode = 'markers')
    fig.show()

x_data = t.tensor([-3, 0.5, 2.5, 3.5])
y_data = t.tensor([1, 0.5, 4, 5])
n_models = 20
hidden_size = 200
n_steps = 200

neural_networks = [NeuralNetwork(hidden_size) for _ in range(n_models)]
original_model_preds = [neural_network(x_data.unsqueeze(0).T) for neural_network in neural_networks]
optimizers = [SGD(nn.parameters(), lr = 0.001) for nn in neural_networks]
print(f'Training {len(neural_networks)} neural networks')
all_ys, weights, jacobian = training_loop(neural_networks, original_model_preds, optimizers, x_data, y_data, steps = n_steps)

plot_weights(weights)
plot_model_functions(all_ys, x_data, y_data)
px.imshow(jacobian.squeeze(2).squeeze(2).T.detach())


