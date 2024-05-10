import torch
from torch import nn


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, shape, device, generator):
        super(VAEModel, self).__init__()
        self.device = device
        self.shape = shape
        self.generator = generator
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latents_shape = (1, 4, self.shape[1] // 8, self.shape[2] // 8)
        encoder_noise = torch.randn(latents_shape, generator=self.generator, device=self.device)

        z = self.encoder(x, encoder_noise)
        y = self.decoder(z)

        return y
