import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# ===== PATHS ======
TEST_DIR='data/combined/test'
MODEL_PATH='models/model.h5'
OUTPUT_DIR='outputs/metrics'

os.makedirs('outputs/metrics', exist_ok=True)

# ===== LOAD TEST DATA ======
test_datagen = ImageDataGenerator (rescale = 1/.255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False
)

# ===== LOAD MODEL AND PREDICT ======
model = load_model(MODEL_PATH)
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# ===== SAVE CLASS INDICES ======
class_labels = test_generator.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_labels, f)

# ====== Accuracy and Loss ======
train_log = pd.read_csv('outputs/logs/train-logs.csv')

train_accuracy = train_log['accuracy']
val_accuracy = train_log['val_accuracy']

train_loss = train_log['loss']
val_loss = train_log['val_loss']

fig, axs = plt.subplots(1, 2)

axs[0].plot(train_accuracy, label='Train Accuracy')
axs[0].plot(val_accuracy, label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(train_loss, label='Train Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('outputs/metrics/accuracy_and_loss_v1.png')
plt.close()
# ======= Confusion Matrix ========
cm = confusion_matrix(y_true, y_pred_classes)
labels = list(class_labels.keys())
sns.heatmap(cm, annot=True, cmap='Greens', xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.tight_layout()

# Save Confusion Matrix
plt.savefig('outputs/metrics/confusion_matrix_v1.png')
plt.close()

# Save precision, accuracy, recall and F1 score to json file
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

report_dict = classification_report(y_true, y_pred_classes, output_dict=True)

result = {
    "model": "model_v1",
    "timestamp": datetime.now().isoformat(timespec='seconds'),
    "metrics": {
        "overall": {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
        },
        "per_class": {}
    }
}

# Input metrics per-class
index_to_label = {v: k for k, v in class_labels.items()}

for idx, class_name in index_to_label.items():
    class_metrics = report_dict.get(str(idx))
    if class_metrics:
        result["metrics"]["per_class"][class_name] = {
            "precision": round(class_metrics["precision"], 2),
            "recall": round(class_metrics["recall"], 2),
            "f1_score": round(class_metrics["f1-score"], 2)
        }

with open(os.path.join(OUTPUT_DIR, 'classification_report_v1.json'), 'w') as f:
    json.dump(result, f, indent=4)

print("Evaluation results saved to 'outputs/metrics/'")