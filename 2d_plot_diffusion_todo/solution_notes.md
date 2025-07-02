# Solution Notes

## Task 1: Simple DDPM pipeline with Swiss-Roll

To implement the `SimpleNet` class, we need to build a noise estimating network that takes the noisy data \( $x_t$ \) and the current diffusion time step \( $t$ \). We'll use the `TimeLinear` class to incorporate the time-dependent transformation.

Here's how you can implement the `SimpleNet` class:

1. **Initialization (`__init__` method)**:
   - Initialize the input and output dimensions, hidden dimensions, and the number of timesteps.
   - Create a `TimeLinear` layer for each hidden layer.
   - Create a final `TimeLinear` layer to map to the output dimension.

2. **Forward Pass (`forward` method)**:
   - Apply the `TimeLinear` layers sequentially to the input \( x \) and the time step \( t \).
   - Return the final output.

Here's the implementation:

```python 2d_plot_diffusion_todo/network.py
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        return alpha * x

class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hids = dim_hids
        self.num_timesteps = num_timesteps

        layers = []
        current_dim = dim_in
        for dim_hid in dim_hids:
            layers.append(TimeLinear(current_dim, dim_hid, num_timesteps))
            layers.append(nn.ReLU())
            current_dim = dim_hid

        layers.append(TimeLinear(current_dim, dim_out, num_timesteps))

        self.network = nn.Sequential(*layers)

        ######################

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        for layer in self.network:
            if isinstance(layer, TimeLinear):
                x = layer(x, t)
            else: # ReLU layer
                x = layer(x)

        ######################
        return x
```

### Explanation of Changes:
1. **Initialization (`__init__` method)**:
   - Created a list of layers (`layers`) to build the network.
   - Added `TimeLinear` layers with ReLU activations for each hidden dimension.
   - Added a final `TimeLinear` layer to map to the output dimension.

2. **Forward Pass (`forward` method)**:
   - Sequentially applied each layer in the network.
   - For `TimeLinear` layers, passed both the input `x` and the time step `t`.
   - For other layers (e.g., ReLU), only passed the input `x`.

This implementation ensures that the network can estimate the noise in the input data \( $x_t$ \) at the given time step \( $t$ \).