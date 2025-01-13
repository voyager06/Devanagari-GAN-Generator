# Devanagari GAN Generator

A Generative Adversarial Network (GAN) implementation for generating Devanagari script characters using TensorFlow.

## Overview

This project implements a GAN to generate synthetic Devanagari script characters. The model consists of a generator that creates images from random noise, and a discriminator that learns to distinguish between real and generated images.

## Features

- Generates 28x28 grayscale Devanagari character images
- Uses modern GAN architecture with convolutional layers
- Includes image preprocessing and data augmentation
- Supports checkpointing for model persistence
- Creates GIF animations to visualize training progress

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL
- ImageIO
- Scikit-learn

## Installation

```bash
pip install tensorflow numpy matplotlib pillow imageio scikit-learn
```

## Project Structure

- `load_and_preprocess_images()`: Handles data loading and preprocessing
- `make_generator_model()`: Defines the generator architecture
- `make_discriminator_model()`: Defines the discriminator architecture
- Training utilities:
  - Loss functions for both generator and discriminator
  - Training step function
  - Image generation and visualization functions

## Model Architecture

### Generator
- Input: 100-dimensional noise vector
- Architecture:
  1. Dense layer (7×7×256)
  2. Multiple transposed convolutional layers
  3. Output: 28×28×1 image with tanh activation

### Discriminator
- Input: 28×28×1 image
- Architecture:
  1. Convolutional layers with LeakyReLU activation
  2. Dropout layers for regularization
  3. Output: Single value prediction (real/fake)

## Usage

1. Prepare your dataset:
```python
train_dataset, test_dataset = load_and_preprocess_images(
    directory_path="path/to/dataset",
    image_size=(28, 28),
    batch_size=32,
    test_size=0.2
)
```

2. Train the model:
```python
# Set training parameters
EPOCHS = 100
noise_dim = 100
batch_size = 256

# Start training
train(train_dataset, EPOCHS)
```

3. Generate images:
```python
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
```

## Training Process

The training process includes:
- Alternating training of generator and discriminator
- Periodic checkpointing every 15 epochs
- Generation of sample images at each epoch
- Creation of an animation showing training progress

## Output Visualization

The training progress is visualized through:
- Sample images generated at each epoch
- A GIF animation showing the evolution of generated images
- Sample outputs saved in PNG format

## Checkpointing

Models are automatically saved during training:
- Checkpoint directory: './training_checkpoints'
- Saved every 15 epochs
- Includes both generator and discriminator states

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
