import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

TEST_DIR='data/combined/test'
MODEL_PATH='models/model.h5'

test_datagen = ImageDataGenerator (rescale = 1/.255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False
)

# Class Prediction
model = load_model(MODEL_PATH)
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

class_labels = test_generator.class_indices

with open("class_indices.json", "w") as f:
    json.dump(class_labels, f)

sns.heatmap(cm, annot=True, cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')

plt.show()