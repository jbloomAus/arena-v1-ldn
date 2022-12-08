import uuid
import argparse
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from GAN_implementation.fun_run_naming import get_random_name

from VAE_implementation.train import train_vae
from VAE_implementation.models import VAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim_size", type=int, default=5)
    parser.add_argument("--mnist_path", type=str, default="data")
    # parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epoch_duration", type=int, default=60*60*2)
    parser.add_argument("--log_image_output_interval", type=int, default=100)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fun_run_names", type=int, default=1)
    args = parser.parse_args()

    # 
    if args.fun_run_names:
        name = get_random_name()
    else:
        name = uuid.uuid4()

    print("Beginning DC GAN training: ", name)

    if args.use_wandb:
        wandb.init(project="VAE_implementation_Joseph", entity="arena-ldn", config=args, name=name)

    train_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = VAE(latent_dim_size=args.latent_dim_size)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_vae(
        autoencoder=model,
        trainloader=train_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        kl_weight=args.kl_weight,
        max_epoch_duration=args.max_epoch_duration,
        log_image_output_interval=args.log_image_output_interval,
        use_wandb=args.use_wandb,
        device=args.device)
    
    if args.use_wandb:
        wandb.finish()