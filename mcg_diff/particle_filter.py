from typing import Tuple, List

import torch
from torch.distributions import Categorical

from mcg_diff.sgm import ScoreModel, generate_coefficients_ddim
from mcg_diff.utils import get_taus_from_singular_values


def predict(score_model: ScoreModel,
            particles: torch.Tensor,
            t: float,
            t_prev: float,
            eta: float,
            n_samples_per_gpu: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise, coeff_sample, coeff_score = generate_coefficients_ddim(
        alphas_cumprod=score_model.alphas_cumprod.to(particles.device),
        time_step=t,
        prev_time_step=t_prev,
        eta=eta
    )
    if hasattr(score_model.net, 'device_ids'):
        batch_size = n_samples_per_gpu * len(score_model.net.device_ids)
        epsilon_predicted = []
        n_batches = particles.shape[0] // batch_size + int(particles.shape[0] % batch_size > 0)
        for batch_idx in range(n_batches):
            epsilon_predicted.append(score_model.net(particles[batch_size*batch_idx:(batch_idx+1)*batch_size], t).cpu())
        epsilon_predicted = torch.cat(epsilon_predicted, dim=0).to(particles.device)
    else:
        epsilon_predicted = score_model.net(particles, t).to(particles.device)
    mean = coeff_sample * particles + coeff_score * epsilon_predicted.to(particles.device)

    return mean, noise, epsilon_predicted


def gauss_loglik(x, mean, diag_std):
    return - 1/2 * (torch.linalg.norm((x - mean[None, :]) / diag_std[None].clip(1e-10, 1e10), dim=-1)**2)


def mcg_diff(
        initial_particles: torch.Tensor,
        observation: torch.Tensor,
        score_model: ScoreModel,
        coordinates_mask: torch.Tensor,
        timesteps: torch.Tensor,
        likelihood_diagonal: torch.Tensor,
        var_observation: float,
        eta: float = 1,
        n_samples_per_gpu_inference: int = 16,
        gaussian_var: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    MCG Diff algorithm, as described in https://arxiv.org/abs/2308.07983
    :param initial_particles: The initial particles for the algorithm
    :param observation: The observation from which we want to sample from the associated posterior
    :param score_model: The score model, containing the score function as well as the alphas_cumprod (VP framework)
    :param coordinates_mask: A mask containing true if the coordinate is observed (corresponds to an observation)
    :param timesteps: The timesteps to be used for the diffusion generation
    :param likelihood_diagonal: The elements of S, such that s_i x_i + var_observations * epsilon_i = y_i
    :param var_observation: the observation variance.
    :param eta: DDIM parameter
    :param n_samples_per_gpu_inference:
    :param gaussian_var: Corresponds to Kappa in https://arxiv.org/abs/2308.07983
    :return: Samples and Log weights.
    '''
    #Initialization
    n_particles, dim = initial_particles.shape
    alphas_cumprod = score_model.alphas_cumprod.to(initial_particles.device)
    particles = initial_particles
    taus, taus_indices = get_taus_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                       timesteps=timesteps,
                                                       singular_values=likelihood_diagonal,
                                                       var=var_observation)

    coordinates_in_state = torch.where(coordinates_mask == 1)[0]
    always_free_coordinates = torch.where(coordinates_mask == 0)[0]
    rescaled_observations = ((alphas_cumprod[taus]**.5)*observation / likelihood_diagonal)

    #Splitting timesteps at after Tau_1 and before tau_1
    filtering_timesteps = timesteps[taus_indices.min().item():]
    propagation_timesteps = timesteps[:taus_indices.min().item()+1]

    pbar = enumerate(zip(filtering_timesteps.tolist()[1:][::-1],
                         filtering_timesteps.tolist()[:-1][::-1]))

    for i, (t, t_prev) in pbar:
        predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                       particles=particles,
                                                       t=t,
                                                       t_prev=t_prev,
                                                       eta=eta,
                                                       n_samples_per_gpu=n_samples_per_gpu_inference)
        active_coordinates_in_obs = torch.where(t_prev >= taus)[0]
        previously_active_coordinates_in_obs = torch.where(t >= taus)[0]
        active_coordinates_in_x = coordinates_in_state[active_coordinates_in_obs]
        inactive_coordinates_in_x = torch.cat((coordinates_in_state[t_prev < taus], always_free_coordinates), dim=0)
        previously_active_coordinates_in_x = coordinates_in_state[previously_active_coordinates_in_obs]

        #Calculation of weights
        previous_log_likelihood = gauss_loglik(
            x=particles[:, previously_active_coordinates_in_x],
            mean=rescaled_observations[previously_active_coordinates_in_obs] * (alphas_cumprod[t] / alphas_cumprod[taus[previously_active_coordinates_in_obs]])**.5,
            diag_std=(1 - (1 - gaussian_var) * (alphas_cumprod[t] / alphas_cumprod[taus[previously_active_coordinates_in_obs]]))**.5)
        log_integration_constant = gauss_loglik(
            x=predicted_mean[:, active_coordinates_in_x],
            mean=rescaled_observations[active_coordinates_in_obs] * ((alphas_cumprod[t_prev] / alphas_cumprod[taus[active_coordinates_in_obs]])**.5),
            diag_std=(predicted_noise ** 2 + 1 - (1 - gaussian_var)*(alphas_cumprod[t_prev] / alphas_cumprod[taus[active_coordinates_in_obs]]))**.5
        )
        log_weights = log_integration_constant - previous_log_likelihood

        #Ancestor sampling
        ancestors = Categorical(logits=log_weights, validate_args=False).sample((n_particles,))
        #Update
        z = torch.randn_like(particles)
        Kprev = (predicted_noise**2 / (predicted_noise**2 + 1 - (1 - gaussian_var)*(alphas_cumprod[t_prev] / alphas_cumprod[taus[active_coordinates_in_obs]])).clip(1e-10, 1e10))
        new_particles = particles.clone()
        new_particles[:, inactive_coordinates_in_x] = z[:, inactive_coordinates_in_x] * predicted_noise + predicted_mean[ancestors][:, inactive_coordinates_in_x]
        new_particles[:, active_coordinates_in_x] = Kprev * rescaled_observations[active_coordinates_in_obs][None,:] * ((alphas_cumprod[t_prev] / alphas_cumprod[taus[active_coordinates_in_obs]])**.5) + \
                                   (1 - Kprev)*predicted_mean[ancestors][:, active_coordinates_in_x] + \
                                   ((1 - (1 - gaussian_var)*(alphas_cumprod[t_prev] / alphas_cumprod[taus[active_coordinates_in_obs]]))*Kprev)**.5 * z[:, active_coordinates_in_x]

        particles = new_particles

    t = filtering_timesteps[0]
    previously_active_coordinates_in_obs = torch.where(t >= taus)[0]
    previously_active_coordinates_in_x = coordinates_in_state[previously_active_coordinates_in_obs]
    previous_log_likelihood = gauss_loglik(
        x=particles[:, previously_active_coordinates_in_x],
        mean=rescaled_observations[previously_active_coordinates_in_obs] * (
                alphas_cumprod[t] / alphas_cumprod[taus[previously_active_coordinates_in_obs]]) ** .5,
        diag_std=(1 - (1 - gaussian_var) * (alphas_cumprod[t] / alphas_cumprod[taus[previously_active_coordinates_in_obs]]))**.5)
    if len(propagation_timesteps) > 1:
        # If Tau_1 > 0 we still have to propagate using the diffusion between tau_1 and 0
        pbar = enumerate(zip(propagation_timesteps.tolist()[1:][::-1],
                             propagation_timesteps.tolist()[:-1][::-1]))

        for i, (t, t_prev) in pbar:
            predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                           particles=particles,
                                                           t=t,
                                                           t_prev=t_prev,
                                                           eta=eta,
                                                           n_samples_per_gpu=n_samples_per_gpu_inference)
            z = torch.randn_like(particles)
            particles = z * predicted_noise + predicted_mean
        log_likelihood = gauss_loglik(x=likelihood_diagonal[None, :]*particles[:, coordinates_in_state],
                                      mean=observation,
                                      diag_std=(torch.ones_like(observation)*var_observation)**.5)
        log_weights = log_likelihood - previous_log_likelihood
    else:
        log_weights = -previous_log_likelihood

    return particles, log_weights