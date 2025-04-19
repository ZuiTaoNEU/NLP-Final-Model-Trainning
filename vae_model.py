"""
VAE Architecture Model for Stable Diffusion.
This implements a Variational Autoencoder (VAE) with separate encoder and decoder components
similar to the one used in Stable Diffusion.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_encoder_block(x, filters, downsample=True):
    """
    Create an encoder block with optional downsampling.
    """
    # Residual connection
    skip = x
    
    # First convolution
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    
    # Second convolution with downsampling if needed
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    
    if downsample:
        x = layers.Conv2D(filters, kernel_size=3, strides=2, padding="same")(x)
        skip = layers.Conv2D(filters, kernel_size=1, strides=2, padding="same")(skip)
    else:
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, kernel_size=1, padding="same")(skip)
    
    # Residual connection
    return layers.Add()([x, skip])

def create_decoder_block(x, filters, upsample=True):
    """
    Create a decoder block with optional upsampling.
    """
    # Residual connection
    skip = x
    
    # First convolution
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    
    # Second convolution with upsampling if needed
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    
    if upsample:
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        skip = layers.UpSampling2D()(skip)
        skip = layers.Conv2D(filters, kernel_size=1, padding="same")(skip)
    else:
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, kernel_size=1, padding="same")(skip)
    
    # Residual connection
    return layers.Add()([x, skip])

def create_attention_block(x, filters):
    """
    Create a self-attention block.
    """
    skip = x
    
    # Normalize input
    x = layers.LayerNormalization()(x)
    
    # Generate query, key, value projections
    query = layers.Conv2D(filters, kernel_size=1)(x)
    key = layers.Conv2D(filters, kernel_size=1)(x)
    value = layers.Conv2D(filters, kernel_size=1)(x)
    
    # Reshape for attention
    shape = tf.shape(query)
    batch_size, h, w, c = shape[0], shape[1], shape[2], shape[3]
    
    query = layers.Reshape((h * w, c))(query)
    key = layers.Reshape((h * w, c))(key)
    value = layers.Reshape((h * w, c))(value)
    
    # Compute attention scores
    attention = tf.matmul(query, key, transpose_b=True)
    attention = attention / (filters ** 0.5)  # Scale by sqrt(d_k)
    attention = tf.nn.softmax(attention, axis=-1)
    
    # Apply attention to value
    context = tf.matmul(attention, value)
    context = layers.Reshape((h, w, c))(context)
    
    # Output projection
    output = layers.Conv2D(filters, kernel_size=1)(context)
    
    # Residual connection
    return layers.Add()([skip, output])

def create_vae_encoder(
    img_height, 
    img_width, 
    latent_channels=4,
    channel_multipliers=[1, 2, 4, 4],
    base_channels=128
):
    """
    Create a VAE encoder model similar to the one in Stable Diffusion.
    """
    # Input image
    inputs = keras.Input(shape=(img_height, img_width, 3))
    
    # Initial convolution
    x = layers.Conv2D(base_channels, kernel_size=3, padding="same")(inputs)
    
    # Encoder blocks
    for i, mult in enumerate(channel_multipliers):
        channels = base_channels * mult
        
        # Double resolution blocks
        x = create_encoder_block(x, channels, downsample=False)
        x = create_encoder_block(x, channels, downsample=False)
        
        # Add attention for higher resolutions
        if i > 1:
            x = create_attention_block(x, channels)
        
        # Downsample except for the last block
        if i < len(channel_multipliers) - 1:
            x = create_encoder_block(x, channels, downsample=True)
    
    # Final processing
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    
    # Produce mean and log variance
    x = layers.Conv2D(latent_channels * 2, kernel_size=3, padding="same")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name="vae_encoder")
    
    return model

def create_vae_decoder(
    latent_height,
    latent_width,
    latent_channels=4,
    channel_multipliers=[1, 2, 4, 4],
    base_channels=128
):
    """
    Create a VAE decoder model similar to the one in Stable Diffusion.
    """
    # Input latent
    inputs = keras.Input(shape=(latent_height, latent_width, latent_channels))
    
    # Initial convolution
    x = layers.Conv2D(base_channels * channel_multipliers[-1], kernel_size=3, padding="same")(inputs)
    
    # Decoder blocks
    for i, mult in reversed(list(enumerate(channel_multipliers))):
        channels = base_channels * mult
        
        # Double resolution blocks
        x = create_decoder_block(x, channels, upsample=False)
        x = create_decoder_block(x, channels, upsample=False)
        
        # Add attention for higher resolutions
        if i > 1:
            x = create_attention_block(x, channels)
        
        # Upsample except for the first block
        if i > 0:
            next_channels = base_channels * channel_multipliers[i-1]
            x = create_decoder_block(x, next_channels, upsample=True)
    
    # Final processing
    x = layers.GroupNormalization()(x)
    x = layers.Activation("swish")(x)
    
    # Output image
    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name="vae_decoder")
    
    return model

def create_vae_sampler():
    """
    Create a sampler model that converts mean and log variance to a latent sample.
    """
    # Input mean and log variance (combined)
    inputs = keras.Input(shape=(None, None, 8))  # 4 for mean, 4 for log variance
    
    # Split into mean and log variance
    mean, logvar = tf.split(inputs, 2, axis=-1)
    
    # Clip log variance for stability
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    
    # Convert to standard deviation
    std = tf.exp(0.5 * logvar)
    
    # Sample from normal distribution
    epsilon = tf.random.normal(tf.shape(mean))
    
    # Reparameterization trick: z = mean + std * epsilon
    z = mean + std * epsilon
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=z, name="vae_sampler")
    
    return model

def create_vae_models(img_height, img_width, latent_channels=4):
    """
    Create the complete VAE models (encoder, decoder, and sampler).
    """
    # Calculate latent dimensions (in Stable Diffusion, images are downsampled by a factor of 8)
    latent_height, latent_width = img_height // 8, img_width // 8
    
    # Create encoder, decoder, and sampler
    encoder = create_vae_encoder(img_height, img_width, latent_channels)
    decoder = create_vae_decoder(latent_height, latent_width, latent_channels)
    sampler = create_vae_sampler()
    
    return encoder, decoder, sampler
