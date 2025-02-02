from fastapi import FastAPI, WebSocket
import modal
from PIL import Image
import base64
from pathlib import Path
from gen_ai.loaders import load_taesd
import numpy as np
import math
import os
import io
import json
from torchvision import transforms
from gen_ai.image_sample import run_txt2img
from torch import autocast

from omegaconf import ListConfig, OmegaConf
from gen_ai.image_sample import *

app = FastAPI()


H = 512
W = 512
C = 4
F = 8
is_legacy = False
return_latents = False

def cv2_to_b64(cv2_img):
    """
    Convert a cv2 image to a base64 string
    """
    img = Image.fromarray(cv2_img)

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")  # PNG format for Base64 encoding

    buffer.seek(0)

    # Encode the bytes buffer into a Base64 string
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64



def encode_pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_base64_to_pil(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def prepare_latents(
    prompt, 
    negative_prompt="", 
    steps=25,
    stage2strength=None,
):

    config="gen_ai/configs/sd_2_1.yaml"
    #ckpt_path = "/models/stabilityai/stable-diffusion-2-1-base/v2-1_512-ema-pruned.safetensors"
    ckpt_path = "models/v2-1_512-ema-pruned.safetensors"
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

    force_uc_zero_embeddings = None
    force_cond_zero_embeddings = None
    batch2model_input = None
    return_latents=False
    filter=None
    T=None
    additional_batch_uc_fields=None
    decoding_t=None

    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])



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

                model.prepare_sampling_loop(
                    cond=c,
                    uc=uc,
                    num_steps=steps,
                    shape=shape,
                    sampler=sampler,
                )


    """
    #app.state.taesd = taesd
    """
    return model, sampler, value_dict, num_samples, additional_model_inputs



def on_sample(model, sampler, value_dict, num_samples, additional_model_inputs, prompt=""):

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():

                attn_maps = []


                samples_z = model.sample_one_step(**additional_model_inputs)
                attn_map = get_attn_maps_from_unet(model.model.diffusion_model)
                attn_maps.append(attn_map)
                # get attention maps

                decoding_t = 1

                #unload_model(model.model)
                #unload_model(model.denoiser)

                load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                outputs = []
                outputs.append(samples)

                samples = outputs[0]
                attn_map_obj = attn_maps[0]

                print(samples)

                # basically a list of 64x64 attn maps for each token
                #return out, attn_maps

                all_attn_maps = {}
                for key in attn_map_obj:
                    #_attn_maps = attn_map_obj[key]
                    a = attn_map_obj[key]

                    attn_maps = []
                    #for a in _attn_maps:
                    splitted_prompt = prompt.split(" ")
                    n = len(splitted_prompt)
                    start = 0
                    arrs = []

                    ### for _ in range(1):
                    arr = []
                    for i in range(start,start+n):
                        b = a[..., i+1] / (a[..., i+1].max() + 0.001)
                        arr.append(b.T)
                    start += n
                    arr = np.hstack(arr)
                    arrs.append(arr)
                    ###

                    arrs = np.vstack(arrs).T
                    final_attn_map = (arrs * 255).clip(0, 255).astype(np.uint8)
                    attn_maps.append(final_attn_map)

                    b64_attns = [cv2_to_b64(attn_map) for attn_map in attn_maps]
                    all_attn_maps[key] = b64_attns

                # Convert from PyTorch tensor to PIL Image
                to_pil = transforms.ToPILImage()  # torchvision transform for conversion

                # Convert each image in the batch to PIL
                pil_images = [to_pil(samples[i]) for i in range(samples.shape[0])]

                # Encode each PIL image to base64
                b64_images = [cv2_to_b64(np.array(pil_image)) for pil_image in pil_images]
                return b64_images, all_attn_maps


STEPS = 25


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive JSON data instead of plain text
            data = await websocket.receive_text()
            parsed_data = json.loads(data)  # Convert JSON string to Python dict
            print(f"Received object: {parsed_data}")

            _type = parsed_data.get("type")
            _data = parsed_data.get("data")

            if _type == "prepare_latents":

                model, sampler, value_dict, num_samples, additional_model_inputs = prepare_latents(
                    _data.get("prompt"),
                    negative_prompt=_data.get("negative_prompt", ""),
                    steps=_data.get("steps", STEPS),
                )
                app.state.model = model
                app.state.sampler = sampler
                app.state.value_dict = value_dict
                app.state.num_samples = num_samples
                app.state.additional_model_inputs = additional_model_inputs

                app.state.max_steps = _data.get("steps", STEPS)
                app.state.current_steps = 0
                app.state.prompt = _data.get("prompt")

                response = {"success": True, "data": parsed_data}
                print("Done preparing latents")
                await websocket.send_text(json.dumps(response))  # Send JSON back

            elif _type == "on_sample":
                if app.state.current_steps >= app.state.max_steps:
                    await websocket.send_text(json.dumps({"error": "Max steps reached"}))
                    continue

                b64_images, all_attn_maps = on_sample(
                    app.state.model, 
                    app.state.sampler, 
                    app.state.value_dict, 
                    app.state.num_samples, 
                    app.state.additional_model_inputs,
                    prompt=app.state.prompt
                )
                app.state.current_steps += 1

                response = {"type": "on_sample", "data": {"images": b64_images, "attn_maps": all_attn_maps}}
                print("Done sampling")
                await websocket.send_text(json.dumps(response))  # Send JSON back


        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
