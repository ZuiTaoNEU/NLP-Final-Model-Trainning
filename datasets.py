"""
Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import os
import base64
import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer, CLIPTextModel
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

AUTO = tf.data.AUTOTUNE
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77


class DatasetUtils:
    def __init__(
        self,
        dataset_name: str = "LeroyDyer/image-description_text_to_image_BASE64",
        batch_size: int = 4,
        img_height: int = 256,
        img_width: int = 256,
    ):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        
        # Initialize CLIP tokenizer and encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # Image augmentation
        self.augmenter = tf.keras.Sequential([
            tf.keras.layers.CenterCrop(img_height, img_width),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),  # Scale to [-1, 1]
        ])
        
        # Download and prepare dataset
        self.data_frame = self.load_dataset(dataset_name)

    def load_dataset(self, dataset_name):
        """Load dataset from Hugging Face Hub"""
        print(f"Loading dataset: {dataset_name}")
        # Here you would implement the dataset loading logic
        # For now, let's assume we're getting a parquet or csv file with 'image_base64' and 'caption' columns
        
        # This is a placeholder - implement actual dataset loading logic based on the dataset structure
        try:
            local_path = hf_hub_download(
                repo_id=dataset_name,
                filename="data.csv",  # This might need to be adjusted based on actual file name
                repo_type="dataset"
            )
            df = pd.read_csv(local_path)
            
            # Ensure the dataframe has the expected columns
            required_columns = ['image_base64', 'caption']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must have columns: {required_columns}")
                
            print(f"Successfully loaded dataset with {len(df)} samples")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Return a minimal dataframe for testing
            return pd.DataFrame({
                'image_base64': [''],  # Empty placeholder
                'caption': ['Test caption']
            })

    def decode_base64_image(self, base64_str: str) -> tf.Tensor:
        """Decode base64 string to image tensor"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',', 1)[1]
                
            # Decode base64 to binary
            image_bytes = base64.b64decode(base64_str)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Convert to tensor
            image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
            
            # Resize
            image_tensor = tf.image.resize(image_tensor, [self.img_height, self.img_width])
            
            return image_tensor
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            # Return a blank image in case of error
            return tf.zeros([self.img_height, self.img_width, 3], dtype=tf.float32)

    def process_text(self, caption: str) -> tf.Tensor:
        """Tokenize text and convert to tensor"""
        # Tokenize the text
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=MAX_PROMPT_LENGTH,
            truncation=True,
            return_tensors="tf"
        )
        
        return text_inputs.input_ids

    def process_sample(self, sample):
        """Process a single sample from the dataset"""
        image_base64 = sample['image_base64']
        caption = sample['caption']
        
        # Decode image from base64
        image = self.decode_base64_image(image_base64)
        
        # Tokenize caption
        tokenized_text = self.process_text(caption)
        
        return image, tokenized_text

    def apply_augmentation(self, image_batch, token_batch):
        """Apply augmentation to the image batch"""
        return self.augmenter(image_batch), token_batch

    def run_text_encoder(self, image_batch, token_batch):
        """Run text encoder on tokenized text"""
        # Convert token batch to correct format
        token_batch = tf.cast(token_batch, tf.int32)
        
        # Create position IDs
        position_ids = tf.expand_dims(tf.range(MAX_PROMPT_LENGTH), 0)
        position_ids = tf.tile(position_ids, [tf.shape(token_batch)[0], 1])
        
        # Run text encoder
        encoded_text = self.text_encoder(token_batch, position_ids)
        
        # Get the last hidden state
        encoded_text = encoded_text[0]  # Get the last hidden state
        
        return image_batch, token_batch, encoded_text

    def prepare_dict(self, image_batch, token_batch, encoded_text_batch):
        """Prepare the final dictionary for training"""
        return {
            "images": image_batch,
            "tokens": token_batch,
            "encoded_text": encoded_text_batch,
        }

    def prepare_dataset(self) -> tf.data.Dataset:
        """Prepare the final dataset for training"""
        # Convert dataframe to tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'image_base64': self.data_frame['image_base64'].values,
            'caption': self.data_frame['caption'].values
        })
        
        # Shuffle dataset
        dataset = dataset.shuffle(self.batch_size * 10)
        
        # Map processing function
        dataset = dataset.map(self.process_sample, num_parallel_calls=AUTO)
        
        # Batch
        dataset = dataset.batch(self.batch_size)
        
        # Apply augmentation
        dataset = dataset.map(self.apply_augmentation, num_parallel_calls=AUTO)
        
        # Run text encoder
        dataset = dataset.map(self.run_text_encoder, num_parallel_calls=AUTO)
        
        # Prepare final dict
        dataset = dataset.map(self.prepare_dict, num_parallel_calls=AUTO)
        
        # Prefetch for performance
        return dataset.prefetch(AUTO)
