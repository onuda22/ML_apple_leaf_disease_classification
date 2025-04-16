# Import Library
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Define Model
"""
    df
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