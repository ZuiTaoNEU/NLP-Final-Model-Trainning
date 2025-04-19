"""
Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

# Usage
python finetune.py --dataset_name LeroyDyer/image-description_text_to_image_BASE64
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Import Hugging Face components
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler

# Local imports
from datasets import DatasetUtils
from trainer import Trainer

MAX_PROMPT_LENGTH = 77
CKPT_PREFIX = "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    # Dataset related.
    parser.add_argument("--dataset_name", default="LeroyDyer/image-description_text_to_image_BASE64", 
                        type=str, help="HuggingFace dataset name")
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_width", default=512, type=int)
    # Optimization hyperparameters.
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--ema", default=0.9999, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    # Training hyperparameters.
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    # Others.
    parser.add_argument("--mp", action="store_true", help="Whether to use mixed-precision.")
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        type=str,
        help="Provide a local path to a diffusion model checkpoint in the `h5`"
        " format if you want to start over fine-tuning from this checkpoint.",
    )

    return parser.parse_args()


def run(args):
    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    print("Initializing dataset...")
    data_utils = DatasetUtils(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    training_dataset = data_utils.prepare_dataset()

    print("Initializing components...")
    # Checkpoint path
    ckpt_path = (
        CKPT_PREFIX
        + f"_epochs_{args.num_epochs}"
        + f"_res_{args.img_height}"
        + f"_mp_{args.mp}"
        + ".h5"
    )
    
    # Initialize components
    # 1. CLIP Tokenizer and Text Encoder
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
    # 2. VAE for encoding and decoding images
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        subfolder="vae",
        use_safetensors=True
    )
    vae_encoder = tf.keras.Model(
        inputs=vae.encoder.input,
        outputs=vae.encoder.output,
    )
    vae_decoder = tf.keras.Model(
        inputs=vae.decoder.input,
        outputs=vae.decoder.output,
    )
    
    # 3. U-Net for predicting noise
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_safetensors=True,
    )
    
    # 4. Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        subfolder="scheduler"
    )

    print("Initializing trainer...")
    diffusion_ft_trainer = Trainer(
        diffusion_model=unet,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        clip_tokenizer=clip_tokenizer,
        clip_text_encoder=clip_text_encoder,
        noise_scheduler=noise_scheduler,
        pretrained_ckpt=args.pretrained_ckpt,
        mp=args.mp,
        ema=args.ema,
        max_grad_norm=args.max_grad_norm,
    )

    print("Initializing optimizer...")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )
    if args.mp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    print("Compiling trainer...")
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    print("Training...")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    diffusion_ft_trainer.fit(
        training_dataset, epochs=args.num_epochs, callbacks=[ckpt_callback]
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
