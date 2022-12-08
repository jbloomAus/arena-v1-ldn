import plotly.express as px
from einops import rearrange
import torch as t
import numpy as np
from sklearn.manifold import TSNE

def plot_output_in_latent_space(model, device):
    '''
    Model should have a decoder method.
    '''
    # Choose number of interpolation points, and interpolation range (you might need to adjust these)
    n_points = 11
    interpolation_range = (-10, 10)

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, model.latent_dim_size), device=device)
    x = t.linspace(*interpolation_range, n_points)
    latent_dim_data[:, :, 0] = x.unsqueeze(0)
    latent_dim_data[:, :, 1] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims", title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
        yaxis=dict(tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
    )
    return fig

def plot_test_data_in_latent_space(model, test_loader):
    ''' 
    Model should have an encoder method.
    '''

    # for each data point, generate their latent space representation
    latent_space_data = []
    true_label = []
    for images, label in test_loader:
        latent_space_data.append(model.encoder(images).detach().cpu().numpy())
        true_label.append(label.detach().cpu().numpy())
        
    latent_space_data = np.concatenate(latent_space_data, axis=0)
    true_label = np.concatenate(true_label, axis=0, dtype=np.int)
    true_label_string = [str(i) for i in true_label]

    # plot the latent space data
    fig = px.scatter(x=latent_space_data.T[0], y=latent_space_data.T[1], color=true_label_string)
    return fig


def plot_test_data_in_latent_space_tsne(model, test_loader):
    ''' 
    Model should have an encoder method.
    '''

    # for each data point, generate their latent space representation
    latent_space_data = []
    true_label = []
    for images, label in test_loader:
        latent_space_data.append(model.encoder(images).detach().cpu().numpy())
        true_label.append(label.detach().cpu().numpy())
        
    latent_space_data = np.concatenate(latent_space_data, axis=0)
    true_label = np.concatenate(true_label, axis=0, dtype=np.int)
    true_label_string = [str(i) for i in true_label]



    tsne = TSNE(n_components=2, random_state=0)
    latent_space_data_2d = tsne.fit_transform(latent_space_data)

    # plot the latent space data
    fig = px.scatter(x=latent_space_data_2d.T[0], y=latent_space_data_2d.T[1], color=true_label_string)
    return fig