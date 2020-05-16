import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.conv2 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu((self.conv1(x)))
        out = self.conv2(out)
        out = out * 0.1
        out = out + residual
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size = 128):
        return input.view(input.size(0), size, 1, 1)

class gaussian_resnet_encoder(nn.Module):
    def __init__(self, image_size = 32, z_dim = 256):
        super(gaussian_resnet_encoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvLayer(3, 16, kernel_size = 5, stride = 1),
            nn.PReLU(),
            ConvLayer(16, 32, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ConvLayer(32, 64, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ConvLayer(64, 64, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ResidualBlock(64),
            Flatten()
        )

        self.fc1_1 = nn.Linear((int)(image_size / 8) * (int)(image_size / 8) * 64, z_dim)
        self.fc2_1 = nn.Linear((int)(image_size / 8) * (int)(image_size / 8) * 64, z_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(torch.device("cuda"))
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu = self.fc1_1(h)
        logvar = self.fc2_1(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

class bernoulli_resnet_decoder(nn.Module):
    def __init__(self, image_size = 32, z_dim = 256):
        super(bernoulli_resnet_decoder, self).__init__()
        self.image_size = image_size
        self.z_dim = z_dim
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            ConvLayer(128, 64, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(64, 32, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(32, 16, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(16, 8, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(8, 3, kernel_size = 3, stride=1),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.fc3 = nn.Linear(z_dim, z_dim * 2)
        self.fc4 = nn.Linear(z_dim * 2, (int)(image_size / 8) * (int)(image_size / 8) * 128)

    def forward(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        input = z.view(z.size(0), 128, (int)(self.image_size / 8), (int)(self.image_size / 8))
        out = self.Upsample(input)
        return out
