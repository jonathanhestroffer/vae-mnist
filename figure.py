import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from scipy.spatial.distance import cdist

from model import VAE

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get MNIST test data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)

    # Initialize the model structure
    in_channels, latent_dim = 1, 10
    model = VAE(in_channels, latent_dim).to(device)  # Replace with your actual model class 

    # Load the saved weights
    model.load_state_dict(torch.load("./model.pth", weights_only=True))

    # Set model to evaluation mode
    model.eval()

    # Get latent vectors to corresponding digits, take mean
    for data, labels in testloader:
        data = data.to(device)
        _, mu, logvar = model(data)
        z = mu + torch.exp(0.5*logvar)

        unq_labels = labels.unique()
        latent_vectors = torch.empty((len(unq_labels), latent_dim)).to(device)
        for i, label in enumerate(unq_labels):
            latent_vectors[i,:] = z[labels==label].mean(dim=0).detach()


    digits = [9, 2, 4, 6, 1]
    n_steps = 11

    vectors = latent_vectors[digits,:]

    anchors = [
        (0,0),
        (n_steps-1,0),
        (0,n_steps-1),
        (n_steps-1,n_steps-1),
        (n_steps//2,n_steps//2)
    ]

    interp_vectors = torch.zeros((n_steps, n_steps, latent_dim)).to(device)

    for vec, (i,j) in zip(vectors, anchors):
        interp_vectors[i,j,:] = vec

    # Distance to an anchor point 
    # acts as an inverse scaling factor (latent vector contribution)
    for i in range(n_steps):

        for j in range(n_steps):

            # skip if anchor point
            if (i,j) in anchors:
                continue
            
            # compute distance to anchor points
            # interpolation factor approx = 1/(d**2)
            dists = torch.tensor(cdist([[i,j]], anchors)).to(device)
            dists = dists.to(torch.float32)
            factors = 1/(dists**2)
            factors /= factors.sum()
            interp_vectors[i,j,:] = (factors@vectors).flatten()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10,10))
    interp_image = np.zeros((28*n_steps,28*n_steps))

    # for i, vec in enumerate(interp_vectors):
    for i in range(n_steps):
        for j in range(n_steps):
            vec = interp_vectors[i,j,:]
            x_recon = model.decode(vec.unsqueeze(0))
            x_recon = x_recon.detach().cpu().numpy()
            interp_image[28*i:28*(i+1),28*j:28*(j+1)] = x_recon[0,0,...]

    ax = plt.imshow(interp_image, cmap="grey")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./figures/interpolation.png", dpi=300)