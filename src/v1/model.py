# Import Library
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Define Model
"""
    Builds a transfer learning image classification model using MobileNetV2 as the base.

    Parameters:
    -----------
    input_shape : tuple, default=(224, 224, 3)
        The shape of the input images (height, width, channels).

    num_classes : int, default=3
        Number of output classes for classification.

    dropout_rate : float, default=0.25
        Dropout rate applied after the dense layer to prevent overfitting.

    dense_units : int, default=256
        Number of units in the dense layer before the output layer.

    Returns:
    --------
    model : tf.keras.Model
        A compiled Keras model ready for training.

    Notes:
    ------
    - The base model uses ImageNet pre-trained weights.
    - The base model is frozen by default to retain learned features.
    - The output layer uses softmax activation for multi-class classification.
    - The model is compiled with RMSprop optimizer and includes accuracy & AUC as metrics.
"""
def build_model(
        input_shape=(224, 224, 3),
        num_classes=3,
        dropout_rate=0.25,
        dense_units=256) :
    
    # Define base model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )

    # Freeze feature extractor
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax', name='classification')
    ])

    model.compile(optimizer=RMSprop(learning_rate=(0.0001)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model