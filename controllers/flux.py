import base64
from gen_ai.loaders import load_taesd
import numpy as np
import math
import os
import io
import server
from torchvision import transforms
from PIL import Image

from gen_ai.flux.cli import load_models
from gen_ai.flux.util import save_image
from pathlib import Path

import torch
from controllers.helpers import cv2_to_b64, decode_base64_to_pil

from gen_ai.flux.sampling import unpack
from einops import rearrange



def normalize_attention_map(y):
    """
    Takes in a Tensor y which is shape (1, 2816, 2816) and 
    returns a normalized Tensor with shape (48, 48)
    """
    HW, B = y.shape
    tensor = y.reshape(int(math.sqrt(HW)), int(math.sqrt(HW)), B)

    # Drop the B
    tensor = tensor.mean(dim=-1)

    # Min-Max Normalization
    min_val = tensor.min()
    max_val = tensor.max()
    y = (tensor - min_val) / (max_val - min_val)

    return y



@torch.inference_mode()
def on_flux_single_sample(
    prompt,
    model, 
    inp, 
    timesteps, 
    opts, 
    ae, 
    guidance_vec,
    t_curr,
    t_prev,
    latent_,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    img, img_ids, txt, txt_ids, vec = inp["img"], inp["img_ids"], inp["txt"], inp["txt_ids"], inp["vec"]
    guidance = opts.guidance
    img_cond = None

    img = latent_ if latent_ is not None else img
    torch_device = torch.device(device)
    t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
    pred = model(
        img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=t_vec,
        guidance=guidance_vec,
    )

    img = img + (t_prev - t_curr) * pred

    # ---

    x = img
    final_attn_map_list = []
    for block in model.double_blocks:
        for _map in block._attn_map_list:
            final_attn_map_list.append(_map)
        

    for block in model.single_blocks:
        for _map in block._attn_map_list:
            final_attn_map_list.append(_map)

    # decode latents to pixel space
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    # ---

    chunk_size = 57
    attn_map_list = final_attn_map_list

    # Break attn_map_list into sublists of 57 items
    #attn_map_grouped = [attn_map_list[i:i+chunk_size] for i in range(0, len(attn_map_list), chunk_size)]

    # WHY is it getting the last attentnion mask?

    print("Got the attn map list", len(attn_map_list))

    # the last 57 items 
    sublist = attn_map_list[-chunk_size:]

    splitted_prompt = prompt.split(" ")
    n = len(splitted_prompt)

    print("Got the splitted prompt", splitted_prompt, n)
    start = 0

    # Process each group: Stack, Mean, Convert to List
    stacked_tensor = torch.stack(sublist).mean(dim=0).to(torch.float32)  # Shape (1, 2815, 2815)

    # Get 512 -> 2815, and 0 -> 511
    y = stacked_tensor[:, 512:, :512]


    to_pil = transforms.ToPILImage()  # torchvision transform for conversion

    arrs = []
    for i in range(start,start+n):
        b = y[..., i+1] / (y[..., i+1].max() + 0.001)
        #arr.append(b.T)
        v = normalize_attention_map(b.T)
        v.to(torch.float32)

        #processed_attn_maps.append(v.tolist())
        arrs.append(v)

    #processed_attn_maps.append(arrs)

    # Done extracting attention maps

    # convert to base64 attention maps
    processed_attn_maps = [to_pil(arr) for arr in arrs]
    processed_attn_maps = [cv2_to_b64(np.array(arr)) for arr in processed_attn_maps]

    # Convert output image into base64
    x = x.to(torch.float32)
    x = rearrange(x[0], "c h w -> h w c")
    x = (127.5 * (x + 1.0)).cpu().byte().numpy()

    output = cv2_to_b64(x)

    print("Got the processed attn maps", len(processed_attn_maps))

    return output, img, processed_attn_maps


def get_timesteps(timesteps, index):
    if index < 0 or index >= len(timesteps) - 1:
        raise IndexError("Index out of range for timesteps list")

    t_curr = timesteps[index]
    t_prev = timesteps[index + 1]

    return t_curr, t_prev


STEPS = 40



