# Stable Diffusion Visualizer
Visualize every attention layer in the UNet for each word in the prompt.


https://github.com/user-attachments/assets/45af9de1-035d-4af5-8946-8740cd0daed3


## 1. Setting up the repository
Make sure you have the following prerequisites installed on your system:
- python version 3.10
- nodejs

The following steps will include commands you can run in your terminal. The commands are written for UNIX based systems like MacOS and Linux.

### 1.1 Download the model
First, download the Stable Diffusion 2.1 model into the /models folder. You can download the model from Huggingface [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors).
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
