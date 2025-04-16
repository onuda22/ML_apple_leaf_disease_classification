# Import Library
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Transformation
"""
    Making a generator for training and validation with image augmentation

    Parameter:
    ----------
    train_dir : str
        Path to directory of trains dataset (The data must already be separated into folders based on their labels)
    val_dir : str
        Path to directory of validations dataset (The data must already be separated into folders based on their labels)
    img_size : tuple, default=(224, 224)
        Size of the image that will be resized before being processed by the model
    batch_size : int, default=32
        Number of images that will be processed in one batch

    Return:
    -------
    train_generator : DirectoryIterator
        Data generator for the training process (with augmentation)

    val_generator : DirectoryIterator
        Data generator for the validating process (without augmentation)

"""
def create_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    data_generator = ImageDataGenerator(
        rescale=1/.255,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.2, 1.8],
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.2, 1.8],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1/.255)

    """
        Generate train and validation generator
    """
    train_generator = data_generator.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator