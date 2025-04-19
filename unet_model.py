"""
U-Net Architecture Model for the Diffusion Model.
This is a simplified version of the U-Net architecture used in Stable Diffusion.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def attention_block(x, channels, num_heads=8):
    """Multi-head attention block."""
    skip = x
    
    # Layer normalization
    x = layers.LayerNormalization()(x)
    
    # Reshape for multi-head attention
    batch_size, height, width, channels = x.shape
    x = layers.Reshape((height * width, channels))(x)
    
    # Multi-head attention
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=channels // num_heads,
        value_dim=channels // num_heads,
    )(x, x, x)
    
    # Reshape back
    x = layers.Reshape((height, width, channels))(x)
    
    # Residual connection
    return layers.Add()([skip, x])

def cross_attention_block(x, context, num_heads=8):
    """Cross-attention block that allows the model to attend to text embeddings."""
    skip = x
    
    # Layer normalization
    x = layers.LayerNormalization()(x)
    context = layers.LayerNormalization()(context)
    
    # Reshape for cross-attention
    batch_size, height, width, channels = x.shape
    x_flattened = layers.Reshape((height * width, channels))(x)
    
    # Cross-attention
    x_flattened = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=channels // num_heads,
        value_dim=channels // num_heads,
    )(x_flattened, context, context)
    
    # Reshape back
    x = layers.Reshape((height, width, channels))(x_flattened)
    
    # Residual connection
    return layers.Add()([skip, x])

def feed_forward_block(x, expansion_factor=4):
    """Feed-forward block with residual connection."""
    skip = x
    channels = x.shape[-1]
    
    # Layer normalization
    x = layers.LayerNormalization()(x)
    
    # Expand
    x = layers.Conv2D(channels * expansion_factor, kernel_size=1, activation="gelu")(x)
    
    # Project back
    x = layers.Conv2D(channels, kernel_size=1)(x)
    
    # Residual connection
    return layers.Add()([skip, x])

def resnet_block(x, channels, downsample=False, time_emb=None):
    """ResNet block with time embedding conditioning."""
    skip = x
    
    # If downsampling is required
    if downsample:
        skip = layers.Conv2D(channels, kernel_size=1, strides=2)(skip)
    elif x.shape[-1] != channels:
        skip = layers.Conv2D(channels, kernel_size=1)(skip)
    
    # First convolution
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(
        channels, 
        kernel_size=3, 
        strides=2 if downsample else 1, 
        padding="same"
    )(x)
    
    # Add time embedding if provided
    if time_emb is not None:
        # Process time embedding
        time_emb = layers.Dense(channels, activation="swish")(time_emb)
        # Reshape for broadcasting
        time_emb = layers.Reshape((1, 1, channels))(time_emb)
        # Add to feature map
        x = layers.Add()([x, time_emb])
    
    # Second convolution
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(channels, kernel_size=3, padding="same")(x)
    
    # Residual connection
    return layers.Add()([skip, x])

def create_unet_model(
    img_height,
    img_width,
    max_text_length=77,
    base_channels=128,
    attention_levels=[1, 2],
    channel_multipliers=[1, 2, 4, 8],
):
    """
    Create a U-Net model for diffusion.
    
    Args:
        img_height: Height of the input image
        img_width: Width of the input image
        max_text_length: Maximum length of the text embedding sequence
        base_channels: Base channel count for the model
        attention_levels: Which levels of the U-Net should have attention blocks
        channel_multipliers: Channel multiplier for each level of the U-Net
        
    Returns:
        A U-Net model that takes noisy latents, timestep embeddings, and text embeddings as input
        and predicts the noise.
    """
    # Define inputs
    latent_height, latent_width = img_height // 8, img_width // 8
    latent_channels = 4  # VAE latent channels
    
    # Input for noisy latent (image in latent space)
    latent_input = keras.Input(shape=(latent_height, latent_width, latent_channels))
    
    # Input for timestep embedding
    time_input = keras.Input(shape=(320,))  # Embedding dimension
    
    # Input for text embedding
    text_input = keras.Input(shape=(max_text_length, 768))  # CLIP text embedding dimension
    
    # Time embedding processing
    time_emb = layers.Dense(1280, activation="swish")(time_input)
    time_emb = layers.Dense(1280, activation="swish")(time_emb)
    
    # Initial projection of latent
    x = layers.Conv2D(base_channels, kernel_size=3, padding="same")(latent_input)
    
    # Store skip connections for U-Net
    skip_connections = []
    
    # Encoder
    for i, mult in enumerate(channel_multipliers):
        # Current channel dimension
        channels = base_channels * mult
        
        # Two ResNet blocks
        x = resnet_block(x, channels, time_emb=time_emb)
        x = resnet_block(x, channels, time_emb=time_emb)
        
        # Store skip connection
        skip_connections.append(x)
        
        # Add attention if this level should have it
        if i in attention_levels:
            x = attention_block(x, channels)
            
            # Cross attention with text embedding
            batch_size = tf.shape(x)[0]
            x_channels = x.shape[-1]
            
            # Process text embedding for cross-attention
            context = layers.Dense(x_channels)(text_input)
            x = cross_attention_block(x, context)
            
            # Feed-forward block after attention
            x = feed_forward_block(x)
        
        # Downsample except for the last encoder block
        if i < len(channel_multipliers) - 1:
            x = layers.Conv2D(channels, kernel_size=3, strides=2, padding="same")(x)
    
    # Middle block
    x = resnet_block(x, channels, time_emb=time_emb)
    x = attention_block(x, channels)
    
    # Cross attention with text embedding for middle block
    context = layers.Dense(channels)(text_input)
    x = cross_attention_block(x, context)
    
    x = feed_forward_block(x)
    x = resnet_block(x, channels, time_emb=time_emb)
    
    # Decoder
    for i, mult in reversed(list(enumerate(channel_multipliers))):
        # Current channel dimension
        channels = base_channels * mult
        
        # Skip connection
        x = layers.Concatenate()([x, skip_connections.pop()])
        
        # Two ResNet blocks
        x = resnet_block(x, channels, time_emb=time_emb)
        x = resnet_block(x, channels, time_emb=time_emb)
        
        # Add attention if this level should have it
        if i in attention_levels:
            x = attention_block(x, channels)
            
            # Cross attention with text embedding
            context = layers.Dense(channels)(text_input)
            x = cross_attention_block(x, context)
            
            # Feed-forward block after attention
            x = feed_forward_block(x)
        
        # Upsample except for the last decoder block
        if i > 0:
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(base_channels * channel_multipliers[i-1], kernel_size=3, padding="same")(x)
    
    # Final projection to predict noise (3 convolution layers with group norm)
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels, kernel_size=3, padding="same")(x)
    
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(latent_channels, kernel_size=3, padding="same")(x)
    
    # Create model
    model = keras.Model(inputs=[latent_input, time_input, text_input], outputs=x)
    
    return model