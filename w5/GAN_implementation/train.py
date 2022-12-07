import time
from typing import Optional, Union

import torch as t
import torch.nn as nn
import wandb
from tqdm import tqdm

from .models import Discriminator, Generator


def train_generator_discriminator(
        netG: Generator,
        netD: Discriminator,
        optG,
        optD,
        trainloader,
        epochs: int,
        max_epoch_duration: Optional[Union[
            int,
            float]] = None,  # Each epoch terminates after this many seconds
        log_netG_output_interval: Optional[Union[
            int,
            float]] = None,  # Generator output is logged at this frequency
        use_wandb: bool = True,
        device: str = "cpu"):


    latent_dim_size = netG.latent_dim_size
    fixed_noise = t.randn(8, latent_dim_size, device=device)

    start_time = time.time()

    if use_wandb:
        wandb.watch(netG, log="all")
        wandb.watch(netD, log="all")

    examples_seen = 0
    for epoch in range(epochs):
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in pbar:

            
            noise = t.randn(trainloader.batch_size, latent_dim_size, device=device)
            
            # 1. Train the discriminator on real+fake
            optD.zero_grad()
            # 1A. Train on real
            real_images = data[0]
            D_x = netD(real_images)
            fake_images = netG(noise)
            D_G_z1 = netD(fake_images.detach())
            lossD = - (t.log(D_x).mean() + t.log(1 - D_G_z1).mean())
            lossD.backward()
            optD.step()

            # 2. Train the generator (maximize log(D(G(z))))
            optG.zero_grad()
            D_G_z2 = netD(fake_images)
            lossG = - (t.log(D_G_z2).mean())
            lossG.backward()
            optG.step()

            # 3. Log metrics
            pbar.set_description(
                f"Epoch: {epoch}, Iteration: {i}, Mean D loss: {lossD.mean().item():.2f}, Mean G loss: {lossG.mean().item():.2f}"
            )
            examples_seen += real_images.size(0)

            if use_wandb:
                wandb.log({
                    "Debug/D(x)": D_x.mean().item(),
                    "Debug/D(G(z)) 1": D_G_z1.mean().item(),
                    "Debug/D(G(z)) 2": D_G_z2.mean().item(),
                    "Losses/D loss": lossD.item(),
                    "Losses/G loss": lossG.item(),
                    "Metrics/Examples seen": examples_seen
                })

            if i % log_netG_output_interval == 0:
                with t.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
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
        t.save(netG.state_dict(), "netG.pth")
        t.save(netD.state_dict(), "netD.pth")

        print("Training complete")

    return netG, netD
