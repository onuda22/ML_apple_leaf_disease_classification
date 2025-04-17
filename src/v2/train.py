import time
import tensorflow as tf

from preprocessing import create_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

TRAIN_DIR='data/combined/train'
VAL_DIR='data/combined/val'
MODEL_PATH='models/model_v2.h5'
LOG_PATH='outputs/logs/train-logs_v2.csv'

MODEL_PATH_FINETUNED='models/model_v2_finetuned.h5'
LOG_PATH_FINETUNED='outputs/logs/train-logs_v2_finetuned.csv'

# Load Data
train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

# Build Model
model, base_model = build_model()

model.compile(optimizer=RMSprop(learning_rate=(1e-4)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

# Setup Checkpoint
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
csv_logger = CSVLogger(LOG_PATH)
earlyStoping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduceLRP = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
callbacks = [checkpoint, csv_logger, earlyStoping, reduceLRP]

# Train Model
start_time = time.time()
model.fit(train_gen,
          validation_data=val_gen,
          epochs=30,
          callbacks=callbacks)
end_time = time.time()

# Calculate time of training
training_time = end_time - start_time

m, s = divmod(training_time, 60)
h, m = divmod(m, 60)
training_time_format = "%d:%02d:%02d" % (h, m, s)

print("Training time 1st Tuning: %s" % training_time_format)

# Fine-Tuning Step 2
# Unfreeze feature extractor from base-model
base_model.trainable = True

# Fine tuned top 40 layer
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Compile with smaller learning rate
model.compile(optimizer=RMSprop(learning_rate=(1e-5)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

checkpoint_finetuned = ModelCheckpoint(MODEL_PATH_FINETUNED, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
csv_logger_finetuned = CSVLogger(LOG_PATH_FINETUNED)
callbacks_finetuned = [checkpoint_finetuned, csv_logger_finetuned, earlyStoping, reduceLRP]

# Continue Training
start_time = time.time()
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_finetuned
)
end_time = time.time()
training_time = end_time - start_time
print("Training time 2nd Tuning: %s" % training_time_format)