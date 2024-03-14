from dataclasses import dataclass
from math import log
from typing import List, Tuple

import torch
from torch import device


@dataclass
class ScoreModel:
    net: torch.nn.Module
    alphas_cumprod: torch.tensor
    device: device

    def to(self, device):
        self.model = self.net.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.device=device

    def cpu(self):
        self.to('cpu')
        self.device = device('cpu')

    def cuda(self):
        self.to('cuda:0')
        self.device = device('cuda:0')


def generate_coefficients_ddim(
        alphas_cumprod,
        time_step,
        prev_time_step,
        eta):
    alphas_cumprod_t_1 = alphas_cumprod[prev_time_step] if prev_time_step >= 0 else 1
    alphas_cumprod_t = alphas_cumprod[time_step]

    noise = eta * (((1 - alphas_cumprod_t_1) / (1 - alphas_cumprod_t)) * (1 - alphas_cumprod_t / alphas_cumprod_t_1)) ** .5

    coeff_sample = (alphas_cumprod_t_1 / alphas_cumprod_t) ** .5
    coeff_score = ((1 - alphas_cumprod_t_1 - noise ** 2) ** .5) - coeff_sample * ((1 - alphas_cumprod_t)**.5)

    return noise, coeff_sample, coeff_score


def ddim_marginal_logprob(
        x0: torch.Tensor,
        alphas_cumprod: List[float],
        timesteps: List[int],
        score_model: ScoreModel,
        n_samples: int,
        eta: float = 1) -> torch.Tensor:
    """
    Computes the log marginal of x0 sampled from ddim.

    steps: 1- sample a path from the real backward process
    conditionned on x0, see eq. (7)
    and compute its logprob 
    2- compute the logprob of the same path under the ddim path log_prob

    output: 
    :log_weights: log ratio, which corresponds to the estimate of the log marginal when
    one sample is used 
    :bwd: forward samples of DDIM, conditionned on the real x0
    """
    dim_range = tuple(range(2,x0.dim() + 1))
    alpha_T = alphas_cumprod[-1]
    noise_sample = torch.randn((n_samples, *x0.shape))
    x = (alpha_T ** .5) * x0 + (1 - alpha_T) ** .5 * noise_sample
    log_weights = ((noise_sample ** 2).sum(dim_range) / 2) - (x**2).sum(dim_range) / 2
    for prev_time_step, time_step in tqdm.tqdm(zip(timesteps[1:],
                                                   timesteps[:-1])):
        alphas_cumprod_t_1 = alphas_cumprod[prev_time_step] if prev_time_step >= 0 else 1
        alphas_cumprod_t = alphas_cumprod[time_step]
        noise_std, coeff_sample, coeff_score = generate_coefficients_ddim(
            alphas_cumprod=score_model.alphas_cumprod,
            time_step=time_step,
            prev_time_step=prev_time_step,
            eta=eta
        )
        epsilon_predicted = score_model.net(x, time_step)
        mean = coeff_sample * x + coeff_score * epsilon_predicted
        if prev_time_step != 0:
            x = (alphas_cumprod_t_1 ** .5) * x0 \
                + (1 - alphas_cumprod_t_1 - noise_std ** 2)**.5 \
                * (x - (alphas_cumprod_t ** .5) * x0) / ((1 - alphas_cumprod_t) ** .5)
            noise_sample = torch.randn_like(x)
            x += noise_std * noise_sample
            log_prob_ddim = - ((x - mean)**2).sum(dim_range) / (2 * noise_std**2)
            log_prob_fwd_ddim = - (noise_sample ** 2).sum(dim_range) / 2
            log_weights += log_prob_ddim - log_prob_fwd_ddim
        else:
            log_prob_ddim = - ((x0 - mean)**2).sum(dim_range) / (2 * noise_std**2)
            log_weights += log_prob_ddim
    return log_weights.logsumexp(0) - log(n_samples)


def ddim_parameters(x: torch.Tensor,
                    score_model: ScoreModel,
                    t: float,
                    t_prev: float,
                    eta: float,) -> Tuple[torch.Tensor, torch.Tensor]:
    noise, coeff_sample, coeff_score = generate_coefficients_ddim(
        alphas_cumprod=score_model.alphas_cumprod.to(x.device),
        time_step=t,
        prev_time_step=t_prev,
        eta=eta
    )
    epsilon_predicted = score_model.net(x, t)
    mean = coeff_sample * x + coeff_score * epsilon_predicted.to(x.device)

    return mean, noise

def ddim_sampling(initial_noise_sample: torch.Tensor,
                  timesteps: List[int],
                  score_model: ScoreModel,
                  eta: float = 1) -> torch.Tensor:
    '''
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    '''
    sample = initial_noise_sample
    for prev_time_step, time_step in zip(timesteps[1:],
                                         timesteps[:-1]):
        mean, noise = ddim_parameters(x=sample,
                                      score_model=score_model,
                                      t=time_step,
                                      t_prev=prev_time_step,
                                      eta=eta)
        sample = mean + noise * torch.randn_like(mean)
    return sample

def ddim_trajectory(initial_noise_sample: torch.Tensor,
                    timesteps: List[int],
                    score_model: ScoreModel,
                    eta: float = 1) -> torch.Tensor:
    '''
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    '''
    sample = initial_noise_sample
    samples = sample.unsqueeze(0)
    for prev_time_step, time_step in zip(timesteps[1:],
                                         timesteps[:-1]):
        mean, noise = ddim_parameters(x=sample,
                                      score_model=score_model,
                                      t=time_step,
                                      t_prev=prev_time_step,
                                      eta=eta)
        sample = mean + noise * torch.randn_like(mean)
        samples = torch.cat([samples, sample.unsqueeze(0)])
    return samples