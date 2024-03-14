from mcg_diff.particle_filter import mcg_diff
from mcg_diff.sgm import ScoreModel
from functools import partial
import torch


def test_particle_filter_inpainting():
    beta_min = 0.1
    beta_max = 30
    beta_d = beta_max - beta_min
    t=torch.linspace(0, 1, steps=1000)
    alphas_cumprod = torch.exp(-.5*(beta_max-beta_min)*(t**2) - beta_min*t)
    timesteps = torch.arange(0, 1001, 10)
    timesteps[-1] -= 1
    samples, lw = mcg_diff(
        initial_particles=torch.randn(size=(100, 2)),
        observation=torch.tensor([0.,]),
        var_observation=0.,
        score_model=ScoreModel(
            net=lambda x, t: ((1 - alphas_cumprod[t])**.5)*x,
            alphas_cumprod=alphas_cumprod,
            device='cpu'
        ),
        likelihood_diagonal=torch.tensor([1.,]),
        coordinates_mask=torch.tensor([True, False]),
        timesteps=timesteps,
        gaussian_var=1e-6,
    )
    assert samples.shape == (100, 2)
    assert (samples[:, 0]**2).max() < 1e-5
    assert lw.shape == (100,)


def test_particle_filter_noisy():
    beta_min = 0.1
    beta_max = 30
    beta_d = beta_max - beta_min
    t=torch.linspace(0, 1, steps=1000)
    alphas_cumprod = torch.exp(-.5*(beta_max-beta_min)*(t**2) - beta_min*t)
    timesteps = torch.arange(0, 1001, 10)
    timesteps[-1] -= 1
    samples, lw = mcg_diff(
        initial_particles=torch.randn(size=(100, 2)),
        observation=torch.tensor([0.,]),
        var_observation=(1 - alphas_cumprod[timesteps[1]]).item(),
        score_model=ScoreModel(
            net=lambda x, t: ((1 - alphas_cumprod[t])**.5)*x,
            alphas_cumprod=alphas_cumprod,
            device='cpu'
        ),
        likelihood_diagonal=torch.tensor([1.,]),
        coordinates_mask=torch.tensor([True, False]),
        timesteps=timesteps,
    )
    assert samples.shape == (100, 2)
    assert lw.shape == (100,)


def test_vmap_particle_filter_inpainting():
    beta_min = 0.1
    beta_max = 30
    beta_d = beta_max - beta_min
    t=torch.linspace(0, 1, steps=1000)
    alphas_cumprod = torch.exp(-.5*(beta_max-beta_min)*(t**2) - beta_min*t)
    timesteps = torch.arange(0, 1001, 10)
    timesteps[-1] -= 1
    samples, lw = torch.func.vmap(mcg_diff, in_dims=(0,), randomness='different')(
        torch.randn(size=(10, 100, 2)),
        observation=torch.tensor([0., ]),
        var_observation=0.,
        score_model=ScoreModel(
            net=lambda x, t: ((1 - alphas_cumprod[t]) ** .5) * x,
            alphas_cumprod=alphas_cumprod,
            device='cpu'
        ),
        likelihood_diagonal=torch.tensor([1., ]),
        coordinates_mask=torch.tensor([True, False]),
        timesteps=timesteps,
        gaussian_var=1e-6
    )
    assert samples.shape == (10, 100, 2)
    assert (samples[:, 0]**2).max() < 1e-5
    assert lw.shape == (10,100,)