# Stable Diffusion Visualizer
Visualize every attention layer in the UNet for each word in the prompt.


https://github.com/user-attachments/assets/45af9de1-035d-4af5-8946-8740cd0daed3


Supported models
- Flux Dev
- Stable Diffusion 2.1


## 1. Setting up the repository
Make sure you have the following prerequisites installed on your system:
- python version 3.10
- nodejs

The following steps will include commands you can run in your terminal. The commands are written for UNIX based systems like MacOS and Linux.

### 1.1 Download the model
** For Flux Dev **
Download the Flux Dev repository into the /models folder. You can download the repository from Huggingface [here](https://huggingface.co/black-forest-labs/FLUX.1-dev). You should have the following files in the /models folder:
- /models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors
- /models/black-forest-labs/FLUX.1-dev/ae.safetensors

You also need to download [OpenAI's CLIP](https://huggingface.co/openai/clip-vit-large-patch14) and [Google's T5 encoders](https://huggingface.co/google/t5-v1_1-xxl) to the models repository.
You should have the following files in the /models folder:
- /models/openai/clip-vit-large-patch14/model.safetensors 
- /models/google/t5-v1_1-xxl/pytorch_model.bin



** For Stable Diffusion 2.1 **
Download the Stable Diffusion 2.1 model into the /models folder. You can download the model from Huggingface [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors).
After you have downloaded the model, the path to the model should be /models/v2-1_512-ema-pruned.safetensors

### 1.2 Install Python server dependencies
Next set up the Python server. In the root of the repository:
- Create a virtual env (optional)
  ```
  python -m venv venv
  source venv/bin/activate
  ```
- Install requirements.txt
  ```
  pip install -r requirements.txt
  ```

Now you can run the Python server with Uvicorn
```
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 1.3 Install frontend dependencies
To set up the frontend, we will need to enter the `web` directory and install packages with npm
```
cd web
npm install
```

Now we are ready to run the app

## 2. Running the app
Boot up the Python server if you haven't already with:
```
uvicorn server:app --host 0.0.0.0 --port 8000
```

In another terminal, enter the frontend and run the start up script like so:
```
cd web
npm run dev
```

Now you are ready to use the app!
