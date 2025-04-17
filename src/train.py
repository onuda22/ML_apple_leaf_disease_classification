import time

from preprocessing import create_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

TRAIN_DIR='data/combined/train'
VAL_DIR='data/combined/val'
MODEL_PATH='models/model.h5'
LOG_PATH='outputs/logs/train-logs.csv'

# Load Data
train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

# Build Model
model = build_model()

# Setup Checkpoint
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
csv_logger = CSVLogger(LOG_PATH)
callbacks = [checkpoint, csv_logger]

# Train Model
start_time = time.time()
model.fit(train_gen,
          validation_data=val_gen,
          epochs=100,
          callbacks=callbacks,
          verbose=1)
end_time = time.time()

# Calculate time of training
training_time = end_time - start_time

m, s = divmod(training_time, 60)
h, m = divmod(m, 60)
training_time_format = "%d:%02d:%02d" % (h, m, s)

print("Training time: %s" % training_time_format)