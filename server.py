from fastapi import FastAPI, WebSocket
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
from PIL import Image

from controllers.sd import prepare_latents, on_sample
from controllers.flux import load_models, on_flux_single_sample, get_timesteps


app = FastAPI()



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    STEPS = 25

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

@app.websocket("/ws-flux")
async def websocket_endpoint(websocket: WebSocket):
    STEPS = 40

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

                prompt = _data.get("prompt")
                num_steps = _data.get("steps", STEPS)

                model, inp, timesteps, opts, ae = load_models(prompt=prompt, num_steps=num_steps)

                app.state.model = model
                app.state.inp = inp
                app.state.timesteps = timesteps
                app.state.opts = opts
                app.state.ae = ae

                app.state.curr_step = 0
                app.state.latent = None

                response = {"success": True, "data": parsed_data}
                print("Done preparing latents")
                await websocket.send_text(json.dumps(response))  # Send JSON back

            elif _type == "on_sample":

                model = app.state.model
                inp = app.state.inp
                timesteps = app.state.timesteps
                opts = app.state.opts
                ae = app.state.ae
                curr_step = app.state.curr_step
                latent = app.state.latent

                if curr_step >= STEPS:
                    await websocket.send_text(json.dumps({"error": "Max steps reached"}))
                    continue

                img, img_ids, txt, txt_ids, vec = inp["img"], inp["img_ids"], inp["txt"], inp["txt_ids"], inp["vec"]
                guidance = opts.guidance
                guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

                t_curr, t_prev = get_timesteps(timesteps, curr_step)
                x, latent, processed_attn_maps = on_flux_single_sample(prompt, model, inp, timesteps, opts, ae, guidance_vec, t_curr, t_prev, latent)


                app.state.latent = latent
                app.state.curr_step += 1


                response = {"type": "on_sample", "data": {"images": [x], "attn_maps": { "0": processed_attn_maps }}}
                print("Done sampling")
                await websocket.send_text(json.dumps(response))  # Send JSON back


        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))


