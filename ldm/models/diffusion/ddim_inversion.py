"""SAMPLING ONLY."""  # TODO: change comment

from random import randrange
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import make_ddim_inversion_sampling_parameters, make_ddim_timesteps


class DDIMInversionSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def set_attribute(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        # ddim sampling parameters
        ddim_alphas, ddim_alphas_next = make_ddim_inversion_sampling_parameters(
            alpha_cum_prod=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            verbose=verbose,
        )
        self.set_attribute("ddim_alphas", ddim_alphas)
        self.set_attribute("ddim_alphas_next", ddim_alphas_next)
        self.set_attribute("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

    @torch.enable_grad()
    def auto_correlation_loss(self, x: torch.Tensor, random_shift=True):
        B, _C, _H, _W = x.shape
        assert B == 1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        regularization_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        for channel_index in range(x.shape[0]):
            noise = x[channel_index][None, None, :, :]
            while True:
                if random_shift:
                    roll_amount = randrange(noise.shape[2] // 2)  # ? what is roll amount?
                else:
                    roll_amount = 1
                regularization_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                regularization_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise: torch.Tensor = F.avg_pool2d(noise, kernel_size=2)
        return regularization_loss

    @torch.enable_grad()
    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var + 1e-7)

    @torch.no_grad()
    def inversion_step(
        self,
        x,
        c,
        t,
        index,
        quantize_denoised=False,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device

        # compute the noise in the image 
        # TODO: use no_grad here
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        else:
            # ? I think this is classifier free guidance
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # TODO: implement regularization of noise prediction here
        lambda_ac = 20
        lambda_kl = 20
        num_reg_steps = 5
        num_ac_rolls = 5

        for _outer in range(num_reg_steps):
            if lambda_ac > 0:
                for _inner in range(num_ac_rolls):
                    _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                    l_ac = self.auto_correlation_loss(_var)
                    _var.retain_grad()
                    l_ac.backward()
                    _grad = _var.grad.detach() / num_ac_rolls
                    e_t = e_t - lambda_ac * _grad
            if lambda_kl > 0:
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_kld = self.kl_divergence(_var)
                l_kld.backward()
                _grad = _var.grad.detach()
                e_t = e_t - lambda_kl * _grad
            e_t = e_t.detach()

        noise_prediction = e_t

        alphas = self.ddim_alphas
        alphas_next = self.ddim_alphas_next
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas_next[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization == "v":
            x0_prediction = self.model.predict_start_from_z_and_v(x, t, model_output)
        else:
            x0_prediction = (x - sqrt_one_minus_at * noise_prediction) / a_t.sqrt()

        if quantize_denoised:
            x0_prediction, _, *_ = self.model.first_stage_model.quantize(x0_prediction)

        direction_pointing_to_xt = (1.0 - a_next).sqrt() * noise_prediction

        # Formula according to https://arxiv.org/abs/2302.03027
        x_next = a_next.sqrt() * x0_prediction + direction_pointing_to_xt

        return x_next, x0_prediction

    @torch.no_grad()
    def invert_latent_image(
        self,
        latent_image,
        conditioning,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
        callback=None,
    ):
        x_latent = latent_image

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        total_steps = timesteps.shape[0]
        print(f"Running DDIM inversion with {total_steps} timesteps")

        iterator = tqdm(timesteps, desc="Decoding image", total=total_steps)  # TODO: change the description
        x_inverted = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_inverted, _ = self.inversion_step(
                x_inverted,
                conditioning,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            if callback:
                callback(i)

        return x_inverted
