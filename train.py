
from tqdm import tqdm

import torch
import torchvision
from torch.optim import Adam
from torchvision import transforms

from model import VAE

def vae_loss(x, x_recon, mu, logvar):
    """Loss function for VAE.

    MSE (reconstruction) + KL-Divergence (regularization)
    """
    recon_loss = torch.nn.functional.mse_loss(x, x_recon, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

if __name__ == "__main__":
    
    # CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transform)

    # Create DataLoaders
    batch_size = 100
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)

    # Initialize model
    model = VAE(in_channels=1, latent_dim=10).to(device)

    optimizer = Adam(params=model.parameters(), lr=5e-4)

    # Train loop
    n_epochs = 15

    for epoch in tqdm(range(n_epochs), desc="Training epochs..."):

        model.train()

        for x, _ in trainloader:

            x = x.to(device)
            optimizer.zero_grad() 

            # forward pass
            x_recon, mu, logvar = model(x)

            # compute loss and backprop
            loss = vae_loss(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model.state_dict(), "./model.pth")