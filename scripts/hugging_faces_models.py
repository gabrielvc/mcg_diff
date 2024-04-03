import os
import math

from diffusers import DDPMPipeline
from scripts.inverse_problems_operators import Deblurring2D, SuperResolution, Inpainting, Colorization
import torch
import numpy as np
from mcg_diff.particle_filter import mcg_diff, ScoreModel
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import PIL.Image
import tqdm


def display_sample(sample):
    image_processed = sample.cpu().permute(1, 2, 0)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_pil)
    #.title(f"Image at step {i}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def display_black_and_white(img):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.imshow(img)
    return fig


def find_furthest_particles_in_clound(particles, N=None):
    N = particles.shape[0]
    dist_matrix = torch.cdist(particles.reshape(N, -1), particles.reshape(N, -1), p=2)
    return (dist_matrix==torch.max(dist_matrix)).nonzero()[0]


class EpsilonNetSVD(torch.nn.Module):

    def __init__(self, H_funcs, unet, dim):
        super().__init__()
        self.unet = unet
        self.H_funcs = H_funcs
        self.dim = dim

    def forward(self, x, t):
        x_normal_basis = self.H_funcs.V(x).reshape(-1, *self.dim)
        #x_normal_basis = x.reshape(-1, 1, 28, 28)
        t_emb = torch.tensor(t).to(x.device)#.repeat(x.shape[0]).to(x.device)
        eps = self.unet(x_normal_basis, t_emb).sample
        #eps_svd_basis = eps.reshape(x.shape[0], -1)
        #eps = eps - .5
        eps_svd_basis = self.H_funcs.Vt(eps, for_H=False)
        return eps_svd_basis


def load_hf_model(config_hf):
    pipeline = DDPMPipeline.from_pretrained(config_hf.hf_model_tag).to('cuda:0')
    all_images = pipeline(batch_size=1)
    image = all_images.images[0]
    x_origin = ((torch.tensor(np.array(image)).type(torch.FloatTensor).cuda() - 127.5) / 127.5)

    D_OR = x_origin.shape
    if len(D_OR) == 2:
        D_OR = (1, ) + D_OR
        x_origin = x_origin.reshape(*D_OR)
    else:
        D_OR = D_OR[::-1]
        x_origin = x_origin.permute(2, 0, 1)
    D_FLAT = math.prod(D_OR)
    return pipeline, x_origin, D_OR, D_FLAT


def plot(x):
    if x.shape[0] == 1:
        fig = display_black_and_white(x[0].cpu())
    else:
        fig = display_sample(x.cpu())
    return fig


def load_operator(task_cfg, D_OR, x_origin):
    sigma_y = task_cfg.sigma_y
    if task_cfg.name == 'deblur_2d':
        kernel_size = math.ceil(D_OR[2] * task_cfg.kernel_size) * (3 // D_OR[0])
        sigma = math.ceil(D_OR[2] * task_cfg.kernel_std)
        pdf = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1 = pdf(torch.arange(-kernel_size, kernel_size + 1)).cuda()
        kernel2 = pdf(torch.arange(-kernel_size, kernel_size + 1)).cuda()
        kernel1 = kernel1 / kernel1.sum()
        kernel2 = kernel2 / kernel2.sum()

        H_funcs = Deblurring2D(kernel1,
                               kernel2,
                               D_OR[0],
                               D_OR[1], 0)


        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0_origin = y_0_origin.reshape(*D_OR)
        y_0 = y_0_origin + sigma_y * torch.randn_like(y_0_origin)
        y_0_img = y_0
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten()[coordinates_mask].cpu()
        diag = diag[coordinates_mask].cpu()
        D_OBS = D_OR

    elif task_cfg.name == 'super_resolution':
        ratio = task_cfg.ratio
        H_funcs = SuperResolution(channels=D_OR[0], img_dim=D_OR[2], ratio=ratio, device='cuda:0')
        D_OBS = (D_OR[0], int(D_OR[1] / ratio), int(D_OR[2] / ratio))
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0_origin = y_0_origin.reshape(*D_OBS)
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = y_0

        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        coordinates_mask = torch.cat(
            (coordinates_mask, torch.tensor([0] * (torch.tensor(D_OR).prod() - len(coordinates_mask))).cuda()))

    elif task_cfg.name == 'outpainting':
        center, width, height = task_cfg.center, task_cfg.width, task_cfg.height
        range_width = (math.floor((center[0] - width / 2)*D_OR[1]), math.ceil((center[0] + width / 2)*D_OR[1]))
        range_height = (math.floor((center[1] - height / 2)*D_OR[2]), math.ceil((center[1] + width / 2)*D_OR[2]))
        mask = torch.ones(*D_OR[1:])
        mask[range_width[0]: range_width[1], range_height[0]:range_height[1]] = 0
        missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

        H_funcs = Inpainting(channels=D_OR[0], img_dim=D_OR[1], missing_indices=missing, device=x_origin.device)
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
        y_0_img[:y_0.shape[-1]] = y_0[0]
        y_0_img = H_funcs.V(y_0_img[None, ...])
        y_0_img = y_0_img.reshape(*D_OR)
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = torch.isin(torch.arange(math.prod(D_OR),
                                                   device=H_funcs.kept_indices.device),
                                      torch.arange(H_funcs.kept_indices.shape[0],
                                                   device=H_funcs.kept_indices.device))
        D_OBS = (math.prod(D_OR) - len(missing),)
    elif task_cfg.name == 'inpainting':
        center, width, height = task_cfg.center, task_cfg.width, task_cfg.height
        range_width = (math.floor((center[0] - width / 2)*D_OR[1]), math.ceil((center[0] + width / 2)*D_OR[1]))
        range_height = (math.floor((center[1] - height / 2)*D_OR[2]), math.ceil((center[1] + width / 2)*D_OR[2]))
        mask = torch.zeros(*D_OR[1:])
        mask[range_width[0]: range_width[1], range_height[0]:range_height[1]] = 1
        missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

        H_funcs = Inpainting(channels=D_OR[0], img_dim=D_OR[1], missing_indices=missing, device=x_origin.device)
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
        y_0_img[:y_0.shape[-1]] = y_0[0]
        y_0_img = H_funcs.V(y_0_img[None, ...])
        y_0_img = y_0_img.reshape(*D_OR)
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = torch.isin(torch.arange(math.prod(D_OR),
                                                   device=H_funcs.kept_indices.device),
                                      torch.arange(H_funcs.kept_indices.shape[0],
                                                   device=H_funcs.kept_indices.device))
        D_OBS = (math.prod(D_OR) - len(missing),)
    elif task_cfg.name == 'colorization':

        H_funcs = Colorization(D_OR[1], x_origin.device)

        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = y_0_origin + sigma_y * torch.randn_like(y_0_origin)
        y_0_img = H_funcs.H_pinv(y_0_origin).reshape(D_OR)
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten()[coordinates_mask].cpu()
        diag = diag[coordinates_mask].cpu()
        coordinates_mask = torch.cat(
            (coordinates_mask, torch.tensor([0] * (torch.tensor(D_OR).prod() - len(coordinates_mask))).cuda()))
        D_OBS = (y_0.shape[-1],)
    else:
        raise NotImplementedError

    return H_funcs, y_0, y_0_origin, y_0_img, U_t_y_0, diag, coordinates_mask, D_OBS


def run_mcg_diff(mcg_diff_config, score_model, n_max_gpu, dim, U_t_y_0, diag, coordinates_mask, sigma_y, timesteps, eta, H_funcs):
    total_N = mcg_diff_config.N_total
    #batch_size = n_max_gpu // mcg_diff_config.N_particles
    n_particles = mcg_diff_config.N_particles
    n_batch = total_N #// batch_size
    def _run(initial_particles):
        particles, weights = mcg_diff(
            initial_particles=initial_particles.cpu(),
            observation=U_t_y_0,
            likelihood_diagonal=diag.cpu(),
            score_model=score_model,
            coordinates_mask=coordinates_mask.cpu(),
            var_observation=sigma_y ** 2,
            timesteps=timesteps.cpu(),
            eta=eta,
            n_samples_per_gpu_inference=n_max_gpu,
            gaussian_var=mcg_diff_config.gaussian_var
        )
        particle = particles[torch.distributions.Categorical(logits=weights, validate_args=True).sample((1,))[0]]
        return particle

    run_fn = _run # would like to do vmap(_run)
    particles_mcg_diff = []
    for j in tqdm.tqdm(enumerate(range(n_batch)), desc="MCG-DIFF"):
        batch_initial_particles = torch.randn(size=(n_particles, dim))
        particles = run_fn(batch_initial_particles)[None]
        H_funcs = H_funcs.to("cpu")
        particles = H_funcs.V(particles).clip(-1, 1)
        H_funcs = H_funcs.to("cuda:0")
        particles_mcg_diff.append(particles)
    particles_mcg_diff = torch.concat(particles_mcg_diff, dim=0)
    return particles_mcg_diff


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    full_path_images = os.path.join(cfg.save_folder,
                                    cfg.task.name,
                                    cfg.dataset.hf_model_tag.replace('-', '_').replace('/','_'),
                                    str(cfg.seed),
                                    'images')
    full_path_data = os.path.join(cfg.save_folder,
                                  cfg.task.name,
                                  cfg.dataset.hf_model_tag.replace('-', '_').replace('/','_'),
                                  str(cfg.seed),
                                  'data')
    Path(full_path_images).mkdir(parents=True, exist_ok=True)
    Path(full_path_data).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    # Loading HF model
    pipeline, x_origin, D_OR, D_FLAT = load_hf_model(cfg.dataset)
    fig = plot(x_origin)
    if cfg.plot:
        fig.show()
    if cfg.save_fig:
        fig.savefig(f'{full_path_images}/sample.pdf')
    plt.close(fig)

    H_funcs, y_0, y_0_origin, y_0_img, U_t_y_0, diag, coordinates_mask, D_OBS = load_operator(task_cfg=cfg.task,
                                                                                              D_OR=D_OR,
                                                                                              x_origin=x_origin)

    fig = plot(y_0_img)
    if cfg.plot:
        fig.show()
    if cfg.save_fig:
        fig.savefig(f'{full_path_images}/measure.pdf')
    plt.close(fig)


    #Diffusion stuff
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.cuda().clip(1e-6, 1)
    timesteps = torch.linspace(0, 999, cfg.diffusion.n_steps).long().cuda()
    eta = cfg.diffusion.eta

    model = pipeline.unet
    model = model.requires_grad_(False)
    model = model.eval()

    ## MCG_DIFF
    particles_mcg_diff = run_mcg_diff(
        mcg_diff_config=cfg.mcg_diff,
        n_max_gpu=cfg.dataset.N_MAX_GPU_MCG_DIFF,
        dim=D_FLAT,
        U_t_y_0=U_t_y_0,
        diag=diag,
        coordinates_mask=coordinates_mask==1,
        sigma_y=cfg.task.sigma_y,
        timesteps=timesteps,
        eta=eta,
        H_funcs=H_funcs,
        score_model=ScoreModel(net=torch.nn.DataParallel(EpsilonNetSVD(H_funcs, model, dim=D_OR).requires_grad_(False)),
                               alphas_cumprod=alphas_cumprod,
                               device='cuda:0'),
    )
    particles_mcg_diff = particles_mcg_diff.reshape(-1, *D_OR)

    furthest = find_furthest_particles_in_clound(particles_mcg_diff)
    for i, particle in enumerate(particles_mcg_diff[furthest]):
        fig = plot(particle)
        if cfg.plot:
            fig.show()
        if cfg.save_fig:
            fig.savefig(f'{full_path_images}/furthest_{i}_mcg_diff.pdf')
        plt.close(fig)
    if cfg.save_data:
        np.save(file=f'{full_path_data}/particles_mcg_diff.npy',
                arr=particles_mcg_diff.cpu().numpy())



    if cfg.save_data:
        np.save(file=f'{full_path_data}/noisy_obs.npy', arr=y_0.cpu().numpy())
        np.save(file=f'{full_path_data}/sample.npy', arr=x_origin.cpu().numpy())
        np.save(file=f'{full_path_data}/noiseless_obs.npy', arr=y_0_origin.cpu().numpy())


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()