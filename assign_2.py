import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

ROLL_NUMBER = 102303803
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20
LATENT_DIM = 10

def load_and_transform_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='cp1252', low_memory=False)
    except:
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    
    data = df['no2'].dropna().values.astype(np.float32)
    
    ar = 0.5 * (ROLL_NUMBER % 7)
    br = 0.3 * ((ROLL_NUMBER % 5) + 1)
    
    print(f"Parameters calculated: ar={ar}, br={br}")
    
    z = data + ar * np.sin(br * data)
    
    mean = np.mean(z)
    std = np.std(z)
    z_normalized = (z - mean) / std
    
    return z_normalized, mean, std, z

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_gan(data_normalized):
    dataset = torch.tensor(data_normalized).view(-1, 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    generator = Generator(LATENT_DIM)
    discriminator = Discriminator()
    
    opt_g = optim.Adam(generator.parameters(), lr=LR)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR)
    
    criterion = nn.BCELoss()
    
    print("Starting GAN training...")
    for epoch in range(EPOCHS):
        for i, real_samples in enumerate(dataloader):
            batch_size = real_samples.size(0)
            
            discriminator.zero_grad()
            
            labels_real = torch.ones(batch_size, 1)
            output_real = discriminator(real_samples)
            loss_d_real = criterion(output_real, labels_real)
            
            noise = torch.randn(batch_size, LATENT_DIM)
            fake_samples = generator(noise)
            labels_fake = torch.zeros(batch_size, 1)
            output_fake = discriminator(fake_samples.detach())
            loss_d_fake = criterion(output_fake, labels_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            
            generator.zero_grad()
            
            labels_g = torch.ones(batch_size, 1)
            output_g = discriminator(fake_samples)
            loss_g = criterion(output_g, labels_g)
            
            loss_g.backward()
            opt_g.step()
            
        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
            
    return generator

def plot_results(generator, real_z, mean, std):
    num_samples = len(real_z)
    noise = torch.randn(num_samples, LATENT_DIM)
    
    with torch.no_grad():
        generated_normalized = generator(noise).numpy().flatten()
    
    generated_data = (generated_normalized * std) + mean
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_z, label='Real Data PDF (z)', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(generated_data, label='GAN Estimated PDF', fill=True, color='red', alpha=0.3)
    plt.title(f'PDF Estimation using GAN\nTransformation Parameters: ar=0.0, br=1.2')
    plt.xlabel('Transformed Variable (z)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    z_norm, mean, std, real_z = load_and_transform_data('/data.csv')
    trained_generator = train_gan(z_norm)
    plot_results(trained_generator, real_z, mean, std)
