from gen_ai.taesd.taesd import TAESD 
import torch


def load_taesd(encoder_path: str = None, decoder_path: str = None) -> TAESD:
    from PIL import Image
    import sys
    import torchvision.transforms.functional as TF
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", dev)
    taesd = TAESD(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    ).to(dev)

    return taesd


    """
    for im_path in sys.argv[1:]:
        im = TF.to_tensor(Image.open(im_path).convert("RGB")).unsqueeze(0).to(dev)

        # encode image, quantize, and save to file
        im_enc = taesd.scale_latents(taesd.encoder(im)).mul_(255).round_().byte()
        enc_path = im_path + ".encoded.png"
        TF.to_pil_image(im_enc[0]).save(enc_path)
        print(f"Encoded {im_path} to {enc_path}")

        # load the saved file, dequantize, and decode
        im_enc = taesd.unscale_latents(TF.to_tensor(Image.open(enc_path)).unsqueeze(0).to(dev))
        im_dec = taesd.decoder(im_enc).clamp(0, 1)
        dec_path = im_path + ".decoded.png"
        print(f"Decoded {enc_path} to {dec_path}")
        TF.to_pil_image(im_dec[0]).save(dec_path)
    """

