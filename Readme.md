# Stable Diffusion Fine-tuning with BASE64 Image Dataset

This project provides a modified version of Stable Diffusion training code to work with BASE64-encoded image datasets, specifically the `LeroyDyer/image-description_text_to_image_BASE64` dataset from Hugging Face.

## Overview

The code implements a full Stable Diffusion pipeline with the following components:

1. **CLIP Tokenizer**: Parses text input and generates token IDs
2. **CLIP Text Encoder**: Converts token IDs into text embeddings
3. **VAE Encoder**: Converts input images into their latent representations
4. **VAE Decoder**: Reconstructs images from the latent space
5. **U-Net Architecture**: Core diffusion model that predicts noise based on noisy latent, timestep, and text condition

## Components

- `datasets.py`: Handles data loading from BASE64-encoded images and preprocessing
- `trainer.py`: Implements the training logic with all components integrated
- `finetune.py`: Main script to run the training process
- `unet_model.py`: Definition of the U-Net architecture
- `vae_model.py`: Definition of the VAE encoder and decoder 
- `clip_model.py`: Implementation of the CLIP tokenizer and text encoder

## Key Changes

The main changes from the original implementation include:

1. Added BASE64 image decoding support to work with the dataset
2. Separated VAE into encoder and decoder components
3. Incorporated CLIP tokenizer and text encoder explicitly
4. Redesigned the U-Net architecture to match Stable Diffusion's approach
5. Updated the training process to integrate all components

## Usage

### Installation

```bash
# Install required packages
pip install tensorflow tensorflow-addons transformers diffusers huggingface_hub pillow
```

### Running the Training

```bash
python finetune.py \
  --dataset_name LeroyDyer/image-description_text_to_image_BASE64 \
  --img_height 512 \
  --img_width 512 \
  --batch_size 4 \
  --num_epochs 100
```

### Additional Parameters

- `--mp`: Enable mixed precision training
- `--pretrained_ckpt`: Path to a pretrained diffusion model checkpoint
- `--lr`: Learning rate (default: 1e-5)
- `--ema`: EMA coefficient for model weights (default: 0.9999)

## How it Works

1. Images are loaded from BASE64 strings and decoded to tensors
2. Text prompts are tokenized and encoded using CLIP
3. Images are encoded to the latent space using the VAE encoder
4. Random noise is added to the latent representations
5. The U-Net model predicts this noise based on the noisy latent, timestep embedding, and text embedding
6. The model is trained to minimize the MSE between the predicted and actual noise

## Notes on the Dataset

The dataset `LeroyDyer/image-description_text_to_image_BASE64` contains image-text pairs where:
- Images are stored as BASE64 encoded strings
- Each image has an associated text description/prompt

## References

This code is adapted from the following sources:
- Hugging Face Diffusers library: https://github.com/huggingface/diffusers
- Keras-CV Stable Diffusion implementation
