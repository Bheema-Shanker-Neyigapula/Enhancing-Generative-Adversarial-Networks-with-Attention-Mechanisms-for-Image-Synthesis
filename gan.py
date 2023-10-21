import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import skimage.metrics
import torch
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import tensorflow_addons as tfa
from skimage.metrics import structural_similarity as ssim

# Generator model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model

# Updated Discriminator model with attention mechanism
def build_discriminator_with_attention(img_shape):
    inputs = layers.Input(shape=img_shape)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Add 1x1 convolution to adjust output shape
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)

    # Apply attention mechanism
    attention = layers.Dense(1, activation='sigmoid')(x)
    attention = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(attention)
    x = layers.Multiply()([x, attention])

    x = layers.GlobalAveragePooling2D()(x)

    validity = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=validity)
    return model

# Combine generator and discriminator into a GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    model = models.Model(gan_input, gan_output)
    return model

# Function to compute Structural Similarity Index (SSIM)
def compute_ssim(real_images, generated_images):
    ssim_scores = []
    for i in range(len(real_images)):
        real_image = tf.squeeze(real_images[i], axis=-1)  # Remove channel axis
        generated_image = tf.squeeze(generated_images[i], axis=-1)  # Remove channel axis

        # Scale the images from [0, 255] to [0, 1] as required by SSIM
        real_image = tf.cast(real_image, tf.float32) / 255.0
        generated_image = tf.cast(generated_image, tf.float32) / 255.0

        # Compute SSIM
        ssim_score = ssim(real_image.numpy(), generated_image.numpy(), data_range=1.0, multichannel=False)
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)

# Function to preprocess and resize images for InceptionV3
def preprocess_resize_images(images, target_shape):
    # Convert images from grayscale to RGB (replicate channel)
    images_rgb = tf.image.grayscale_to_rgb(images)
    # Resize images to the target shape expected by InceptionV3
    images_resized = tf.image.resize(images_rgb, target_shape)
    # Preprocess images as required by InceptionV3
    images_preprocessed = preprocess_input_inception(images_resized)
    return images_preprocessed

# Function to calculate the FID
def calculate_fid(real_images, generated_images, batch_size=50):
    # Convert NumPy arrays to TensorFlow tensors
    real_images = tf.convert_to_tensor(real_images)
    generated_images = tf.convert_to_tensor(generated_images)

    # Resize and preprocess real and generated images for InceptionV3
    target_shape = (299, 299)  # InceptionV3 input shape
    real_images_preprocessed = preprocess_resize_images(real_images, target_shape)
    generated_images_preprocessed = preprocess_resize_images(generated_images, target_shape)

    # Compute the FID using FBetaScore from tensorflow_addons
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
    real_features = inception_model(real_images_preprocessed)
    generated_features = inception_model(generated_images_preprocessed)

    fbeta_score_metric = tfa.metrics.FBetaScore(num_classes=2048, beta=1.0)  # Set beta to 1.0 for FID
    fbeta_score_metric.update_state(real_features, generated_features)
    fid_value = fbeta_score_metric.result().numpy()
    return fid_value

# Function to preprocess generated images for InceptionV3
def preprocess_real_images(images):
    # Replicate the grayscale image along the channel axis to create an RGB image
    return np.repeat(images, 3, axis=-1)

# Function to preprocess generated images for InceptionV3
def preprocess_generated_images(images):
    # Replicate the grayscale image along the channel axis to create an RGB image
    rgb_images = np.repeat(images, 3, axis=-1)

    # Resize images to the expected size of InceptionV3 (299x299)
    resized_images = tf.image.resize(rgb_images, (299, 299))

    return resized_images

# Function to compute the Inception Score
def calculate_inception_score(generated_images, batch_size=32):
    # Preprocess generated images for InceptionV3
    generated_images_preprocessed = preprocess_generated_images(generated_images)

    # Load the pre-trained InceptionV3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    inception_model.trainable = False

    # Calculate Inception Score using InceptionV3 model
    features = inception_model.predict(generated_images_preprocessed, batch_size=batch_size)
    preds = tf.nn.softmax(features)

    # Calculate mean and standard deviation of the softmax predictions
    inception_score_mean = np.mean(preds, axis=0)
    inception_score_std = np.std(preds, axis=0)

    return inception_score_mean, inception_score_std


# Hyperparameters
latent_dim = 100
img_shape = (28, 28, 1)
batch_size = 64

# Build the generator, discriminator, and GAN models
generator = build_generator(latent_dim)
discriminator = build_discriminator_with_attention(img_shape)
gan = build_gan(generator, discriminator)

# Compile models
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize images to [-1, 1]

# Create a shorter dataset
short_dataset_size = 1000
train_images = train_images[:short_dataset_size]

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=short_dataset_size).batch(batch_size, drop_remainder=True)  # Add 'drop_remainder=True'

# Create empty lists to store the metrics after each epoch
inception_scores = []
fid_scores = []
ssim_scores = []

# Training loop
epochs = 5  # Reduce the number of epochs

for epoch in range(epochs):
    for real_imgs in train_dataset:
        # Generate random noise samples as input to the generator
        noise = tf.random.normal(shape=(batch_size, latent_dim))

        # Generate fake images using the generator
        generated_imgs = generator.predict(noise)

        # Create labels for real and generated images
        real_labels = tf.ones((batch_size, 1))
        generated_labels = tf.zeros((batch_size, 1))

        # Train the discriminator with real images
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

        # Train the discriminator with generated images
        d_loss_fake = discriminator.train_on_batch(generated_imgs, generated_labels)

        # Calculate the average discriminator loss
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = tf.random.normal(shape=(batch_size, latent_dim))
        valid_labels = tf.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

    # Compute Inception Score
    inception_score_mean, inception_score_std = calculate_inception_score(generated_imgs)
    inception_scores.append(inception_score_mean)

    # Compute FID
    fid_value = calculate_fid(real_imgs, generated_imgs)
    fid_scores.append(fid_value)

    # Compute SSIM
    ssim_score = compute_ssim(real_imgs, generated_imgs)
    ssim_scores.append(ssim_score)

    # Print the metrics
    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}")
    print(f"Inception Score: {inception_score_mean} (std: {inception_score_std})")
    print(f"FID: {fid_value}")
    print(f"SSIM: {ssim_score}")

# Plot the metrics
plt.figure(figsize=(10, 5))
plt.plot(inception_scores, label="Inception Score")
plt.plot(fid_scores, label="FID")
plt.plot(ssim_scores, label="SSIM")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend(["Inception Score", "FID", "SSIM"])  # Show one legend for all metrics
plt.grid(True)
plt.title("GAN Metrics")
plt.show()