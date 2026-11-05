import torch
from torch import nn
import numpy as np

class buildClassical(nn.Module): # Based on https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/#google_vignette

    def __init__(self, cfg, rngs):
        super().__init__()
        self.cfg = cfg
        self.rngs = rngs
        self.paradigm = cfg.training.paradigm

        if cfg.circuit.activation=="ReLU":
            self.activation = nn.ReLU()
        elif cfg.circuit.activation=="LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif cfg.circuit.activation=="GELU":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be ReLU or LeakyReLU.")
        
        layer_sizes = cfg.circuit.encoder_shape

        ### --- Build encoder layers from consecutive pairs --- ###
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.encoder_layers = nn.ModuleList(encoder_layers)

        ### --- Build decoder layers — mirror of encoder, dims reversed --- ###
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.classifier = nn.Linear(layer_sizes[-1], 1)

    # Encoding, but no activation on latent space
    def encode(self, x: torch.Tensor):
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < len(self.encoder_layers) - 1:
                x = self.activation(x)
        return x

    def decode(self, z: torch.Tensor):
        for i, layer in enumerate(self.decoder_layers):
            x = layer(z if i == 0 else x)
            # No activation on the final output layer
            if i < len(self.decoder_layers) - 1:
                x = self.activation(x) 
        return x

    def forward(self, x):
        z = self.encode(x)

        if self.paradigm == "unsupervised":
            return self.decode(z)
        elif self.paradigm == "supervised":
            return self.classifier(z).squeeze(-1)
        else:
            raise ValueError("unsupervised or supervised paradigms only!")        

    def param_generator(self):
        """
        Generates initial parameters using config settings (as in quantum code)
        """
        gen = self.rngs["circuit"]

        init_type = self.cfg.training.init_weights[0]

        if init_type == "Normal":
            mu = self.cfg.training.init_weights[1]
            sigma = self.cfg.training.init_weights[2]

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mu, sigma, generator=gen)
                    if m.bias is not None:
                        m.bias.data.zero_()

        elif init_type == "Uniform":
            lower = self.cfg.training.init_weights[1]
            upper = self.cfg.training.init_weights[2]

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.uniform_(lower, upper, generator=gen)
                    if m.bias is not None:
                        m.bias.data.zero_()

        elif init_type == "Kaiming":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(
                        m.weight, a=0.01, nonlinearity="leaky_relu",
                        generator=gen
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        else:
            raise ValueError("Parameter generator must be 'Normal', 'Uniform' or 'Kaiming'!")