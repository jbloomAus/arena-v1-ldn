import time
from typing import Optional, Union

import torch as t
import torch.nn as nn
import wandb
from tqdm import tqdm

from .models import VAE

def train_vae(
        autoencoder: VAE,
        trainloader,
        epochs: int,
        optimizer: t.optim.Optimizer,
        kl_weight: float = 0.1,
        max_epoch_duration: Optional[Union[
            int,
            float]] = None,  # Each epoch terminates after this many seconds
        log_image_output_interval: Optional[Union[
            int,
            float]] = None,  # Generator output is logged at this frequency
        use_wandb: bool = True,
        device: str = "cpu"):


    latent_dim_size = autoencoder.latent_dim_size

    start_time = time.time()

    if use_wandb:
        wandb.watch(autoencoder, log="all")

   
    examples_seen = 0
    fixed_noise = t.randn(8, latent_dim_size, device=device)
    
    for epoch in range(epochs):
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, (images, _) in pbar:
            
            autoencoder.train()
            images = images.to(device)
            
            optimizer.zero_grad()
            mu, logsigma, outputs = autoencoder(images)
            MSE = nn.functional.mse_loss(outputs, images)
            KLD = (0.5 * (mu ** 2 + t.exp( logsigma * 2 ) - 1) - logsigma).mean()
            loss = MSE + kl_weight*KLD
            loss.backward()
            optimizer.step()

            examples_seen += mu.size(0)
            time_elapsed = time.time() - start_time

            # 3. Log metrics
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | MSE Loss: {MSE.item():.4f} | KLD Loss: {KLD.item():.4f} | Examples seen: {examples_seen} | Time elapsed: {time_elapsed:.2f}s"
            )
            examples_seen += mu.size(0)

            autoencoder.eval()
            if use_wandb:
                wandb.log({
                    "losses/total": loss.item(),
                    "losses/MSE": MSE.item(),
                    "losses/KLD": KLD.item(),
                    "metrics/examples_seen": examples_seen,
                    "metrics/time_elapsed": time_elapsed,
                    "latents/mean": mu.mean().item(),
                    "latents/std": mu.std().item(),
                })

                if i % log_image_output_interval == 0:
                    with t.no_grad():
                        fake = autoencoder.decoder(fixed_noise).detach().cpu()
                    wandb.log({
                        "images": [
                            wandb.Image(x,
                                        caption="epoch: " + str(epoch) +
                                        " iteration: " + str(i)) for x in fake
                        ]
                    })

            if max_epoch_duration is not None:
                if time.time() - start_time > max_epoch_duration:
                    break
            
        # 4. Save model checkpoints
        t.save(autoencoder.state_dict(), "mnist_autoencoder.pth")

        print("Training complete")

    return autoencoder
