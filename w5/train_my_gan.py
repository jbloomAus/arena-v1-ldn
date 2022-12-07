import uuid
import argparse
import wandb
from torch import optim
from torch.utils.data import DataLoader

from GAN_implementation.train import train_generator_discriminator
from GAN_implementation.data import get_real_image_trainset
from GAN_implementation.models import Generator, Discriminator, initialize_weights_generator, initialize_weights_discriminator
from GAN_implementation.fun_run_naming import get_random_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--latent_dim_size", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--image_path", type=str, default="img_align_celeba")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max_epoch_duration", type=int, default=60*60*2)
    parser.add_argument("--log_netG_output_interval", type=int, default=100)
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
        wandb.init(project="GAN_implementation_Joseph", entity="arena-ldn", config=args, name=name)

    trainset = get_real_image_trainset(args.image_path, args.image_size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    my_Generator = Generator(
        args.latent_dim_size, 
        args.image_size,
        args.img_channels,
        args.hidden_dim,
        args.n_layers
    ).to(args.device)

    initialize_weights_generator(my_Generator)
    
    my_Discriminator = Discriminator(
        args.image_size,
        args.img_channels,
        args.hidden_dim,
        args.n_layers
    ).to(args.device)

    initialize_weights_discriminator(my_Discriminator)

    # # check init means and variance
    # for name, param in my_Generator.named_parameters():
    #     if param.requires_grad:
    #         print(f"Generator    : {name:35} {param.data.mean(): 3.2f} {param.data.std(): 3.6f}")
    
    # for name, param in my_Discriminator.named_parameters():
    #     if param.requires_grad:
    #         print(f"Discriminator: {name:35} {param.data.mean(): 3.2f} {param.data.std(): 3.6f}")

    betas = (args.beta1, args.beta2)
    discriminator_optimizer = optim.Adam(my_Discriminator.parameters(), lr=args.lr, betas=betas)
    generator_optimizer = optim.Adam(my_Generator.parameters(), lr=args.lr, betas=betas)

    my_Generator, my_Discriminator = train_generator_discriminator(
        my_Generator,
        my_Discriminator,
        generator_optimizer,
        discriminator_optimizer,
        trainloader,
        epochs=args.epochs,
        max_epoch_duration=args.max_epoch_duration, # 30 minutes
        log_netG_output_interval=args.log_netG_output_interval,
        use_wandb=args.use_wandb,
        device=args.device,
    )
        