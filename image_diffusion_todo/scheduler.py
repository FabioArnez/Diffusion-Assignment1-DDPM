from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
        device="cuda",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.device = device
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        # Get the required constants for the current timestep
        def extract(input, t: torch.Tensor, x: torch.Tensor):
            if t.ndim == 0:
                t = t.unsqueeze(0)
            shape = x.shape
            t = t.long().to(input.device)
            out = torch.gather(input, 0, t)
            reshape = [t.shape[0]] + [1] * (len(shape) - 1)
            return out.reshape(*reshape)

        sigmas_t = extract(self.sigmas, t, x_t)
        alphas_t = extract(self.alphas, t, x_t)
        alphas_cumprod_t = extract(self.alphas_cumprod, t, x_t)
        z = torch.randn_like(x_t) if t > 0 else 0.0
        # Calculate the coefficients
        coeff1 = torch.sqrt(1 / alphas_t)
        eps_factor = (1 - alphas_t) / torch.sqrt(1 - alphas_cumprod_t)
        # Calculate the mean of the distribution
        mu = coeff1 * (x_t - (eps_factor * eps_theta))
        sample_prev = mu + sigmas_t * z
        #######################
        
        return sample_prev
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        # consts = consts.to('mps')
        # t = t.to('mps') 
        # const = consts.gather(-1, t.to(self.device))
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alpha_cumprod_t = self._get_teeth(self.alphas_cumprod, t)
        x_t = alpha_cumprod_t.sqrt() * x_0 + (1 - alpha_cumprod_t).sqrt() * eps
        #######################

        return x_t, eps
