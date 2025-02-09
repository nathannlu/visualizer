from pytorch_lightning import seed_everything
import copy
import math
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TT
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from .sgm.discretization import (
    Img2ImgDiscretizationWrapper,
    Txt2NoisyDiscretizationWrapper,
)
from gen_ai.sgm.modules.diffusionmodules.guiders import (
    LinearPredictionGuider,
    TrianglePredictionGuider,
    VanillaCFG,
)
from gen_ai.sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from gen_ai.sgm.util import append_dims, default, instantiate_from_config
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid, save_image

def load_model(model):
    model.cuda()

lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()

def load_model_from_config(config, ckpt=None, verbose=True):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                global_step = pl_sd["global_step"]
                print(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError

        msg = None

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        msg = None

    model = initial_model_load(model)
    model.eval()
    return model, msg

def get_batch(
    keys,
    value_dict: dict,
    N: Union[List, ListConfig],
    device: str = "cuda",
    T: int = None,
    additional_batch_uc_fields: List[str] = [],
):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = [value_dict["prompt"]] * math.prod(N)

            batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        elif key == "polars_rad":
            batch[key] = torch.tensor(value_dict["polars_rad"]).to(device).repeat(N[0])
        elif key == "azimuths_rad":
            batch[key] = (
                torch.tensor(value_dict["azimuths_rad"]).to(device).repeat(N[0])
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc


def sample_for_steps(
    _sample_for_steps, 
    model = None, 
    vae = None,
    **additional_model_inputs
):
    """
    Output:

    step | frames
    -------------
    0      [base64_image1, base64_image2, base64_image3]
    1      [base64_image1, base64_image2, base64_image3]
    2      [base64_image1, base64_image2, base64_image3]
    3      [base64_image1, base64_image2, base64_image3]
    """


    from einops import rearrange
    import numpy as np

    with torch.no_grad():
        with torch.autocast("cuda"):
            model.en_and_decode_n_samples_a_time = 1

            output_decoded = {}

            for i in range(_sample_for_steps):
                samples_z = model.sample_one_step(**additional_model_inputs)
                samples_x = decode_first_stage(vae, samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                output_decoded[i] = vid

            # loop over output_decoded and convert each frame to base64
            for key, vid in output_decoded.items():
                base64_images = _convert_frames_to_base64(vid)
                output_decoded[key] = base64_images
            
            return output_decoded


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    outputs = []
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                #unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider,
                            (
                                VanillaCFG,
                                LinearPredictionGuider,
                                TrianglePredictionGuider,
                            ),
                        ):
                            additional_model_inputs[k] = torch.zeros(
                                num_samples[0] * 2, num_samples[1]
                            ).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)

                steps = 25
                model.prepare_sampling_loop(
                    cond=c,
                    uc=uc,
                    num_steps=steps,
                    shape=shape,
                    sampler=sampler,
                )

                attn_maps = []
                for i in range(steps):
                    samples_z = model.sample_one_step(**additional_model_inputs)
                    attn_map = get_attn_maps_from_unet(model.model.diffusion_model)
                    attn_maps.append(attn_map)
                    # get attention maps


                #unload_model(model.model)
                #unload_model(model.denoiser)

                load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                outputs.append(samples)


                if return_latents:
                    return samples, samples_z
                return samples, attn_maps

def get_discretization(discretization, options, key=1, sigma_min=0.03, sigma_max=14.61, rho=3.0):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config


def get_sampler(sampler_name, steps, discretization_config, guider_config, key=1, s_churn=0.0, s_tmin=0.0, s_tmax=999.0, s_noise=1.0, eta=1.0, order=4):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler

def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None, orig_width=512, orig_height=512, crop_coords_top=0, crop_coords_left=0, fps=6, motion_bucket_id=127, mb_id=127, pool_image=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ""

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

        if key in ["fps_id", "fps"]:
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1

        if key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = mb_id

        if key == "pool_image":
            """
            image = load_img(
                key="pool_image_input",
                size=224,
                center_crop=True,
            )
            """
            image = None
            if image is None:
                print("Need an image here")
                image = torch.zeros(1, 3, 224, 224)
            value_dict["pool_image"] = image

    return value_dict

def get_guider(
    options, 
    key,
    guider="VanillaCFG",
    scale=5.0,
    cfg=1.5,
    min_cfg=1.0,
):

    additional_guider_kwargs = options.pop("additional_guider_kwargs", {})

    if guider == "IdentityGuider":
        guider_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        guider_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale,
                **additional_guider_kwargs,
            },
        }
    elif guider == "LinearPredictionGuider":
        max_scale = cfg
        min_scale = min_cfg

        guider_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    elif guider == "TrianglePredictionGuider":
        cfg = 2.5
        max_scale = cfg
        min_scale =  min_cfg
        guider_config = {
            "target": "gen_ai.sgm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    else:
        raise NotImplementedError
    return guider_config



def init_sampling(
    key=1,
    steps=25,
    sampler="EulerEDMSampler",
    discretization="EDMDiscretization",
    img2img_strength: Optional[float] = None,
    specify_num_samples: bool = True,
    stage2strength: Optional[float] = None,
    options: Optional[Dict[str, int]] = None,
):
    options = {} if options is None else options

    num_rows, num_cols = 1, 1

    discretization_config = get_discretization(discretization, options=options, key=key)

    guider_config = get_guider(options=options, key=key)

    sampler = get_sampler(sampler, steps, discretization_config, guider_config, key=key)
    if img2img_strength is not None:
        print(
            f"Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper"
        )
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )
    return sampler, num_rows, num_cols

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_attn_maps_from_unet(unet):
    ###
    # Get the attention map
    ###
    CLIP_SEQ_TOKEN_LENGTH = 77
    TRANSFORMER_KEY = "transformer_blocks.0.attn2" # spatial

    attn_map = {}
    attn_i = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__

        if TRANSFORMER_KEY in name and module_name == "CrossAttention":

            # this one is for spatial transformer
            sim = module._attn_map

            batch, spatial_latent, text_cond = sim.shape
            print("Found spatial latent", spatial_latent)

            attn_key = f"{attn_i}_{spatial_latent}"
            if spatial_latent not in attn_map:
                attn_map[attn_key] = []
            
            attn_map[attn_key].append(sim)
            attn_i += 1


    # different layers goes from
    # - 4096
    # - 1024
    # - 256
    # - 64
    # torch.Size([2, 1024, 77])
    #attn_map = attn_map.get(1024, None)
    for key in attn_map:
        _attn_map = _concat_sum_attn_maps(attn_map[key], dim="batch")
    
        # Loop over all attention maps per token, and 
        # reshape them into a 2D image
        resized_attn_map_output = torch.zeros((64, 64,77)) # this is a 64x64 image for each 77 text token
        f = transforms.Resize(size=(64, 64))
        for i in range(77):
            spatial_latent, text_cond = _attn_map.shape
            current_attn_map = _attn_map[..., i]

            # reshape
            img_dim = int(np.sqrt(spatial_latent)) # this assumes the image has aspect ratio 1:1
            attn_heatmap = current_attn_map.view(1, img_dim, img_dim)
            resized_attn_map_output[..., i] = f(attn_heatmap)[0]

        attn_map[key] = resized_attn_map_output

    return attn_map

def _concat_sum_attn_maps(attn_map, dim="batch"):
    if dim == "batch":
        attn_map = torch.cat(attn_map, dim=0)
        attn_map = attn_map.sum(dim=0)
    elif dim == "tokens":
        attn_map = torch.cat(attn_map, dim=2)
        attn_map = attn_map.sum(dim=2)
    else:
        raise ValueError("Invalid dim")

    return attn_map


def _resize_attn_map(attn_map):
    spatial_latent, text_cond = attn_map.shape
    f = transforms.Resize(size=(64, 64))
    resized_attn_map_output = torch.zeros((64, 64))

    img_dim = int(np.sqrt(spatial_latent)) # this assumes the image has aspect ratio 1:1
    attn_heatmap = current_attn_map.view(1, img_dim, img_dim)


def run_txt2img(
    prompt,
    negative_prompt="",
    H=512,
    W=512,
    C=4,
    F=8,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
):

    config="gen_ai/configs/sd_2_1.yaml"
    ckpt_path = "/models/stabilityai/stable-diffusion-2-1-base/v2-1_512-ema-pruned.safetensors"
    config = OmegaConf.load(config)
    model, msg = load_model_from_config(config, ckpt=ckpt_path)

    print("Loaded model")
    unet = model.model.diffusion_model

    #TRANSFORMER_KEY = "time_stack.0.attn2" # temporal
    TRANSFORMER_KEY = "transformer_blocks.0.attn2" # spatial

    # Turn on attn mapping for attn2
    for name, module in unet.named_modules():
        module_name = type(module).__name__

        # Temporal transformer
        if TRANSFORMER_KEY in name and module_name == "CrossAttention":
            print("Found attn2", module)

            module._is_attn2 = True


    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    sampler, num_rows, num_cols = init_sampling(stage2strength=stage2strength)
    num_samples = num_rows * num_cols
    out, attn_maps = do_sample(
        model,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=return_latents,
        filter=filter,
    )




    # basically a list of 64x64 attn maps for each token
    return out, attn_maps


