# Advancing Infant Distress Detection: Two- and Three-Way Classification in Real-World Audio Environments

## 📌 Model Overview

This repository contains the pre-trained model used in our research paper:
**"Advancing Infant Distress Detection: Two- and Three-Way Classification in Real-World Audio Environments"**

The model is based on **YAMNet + SVM** and is designed for classifying infant distress in **real-world audio environments**. It is optimized for **5-second audio chunks** to improve prediction accuracy.

---

## 📂 Model Structure

The repository includes the following components:

- **`Distress_Binary_Classification_Model.pkl.zip`**: Contains the pre-trained YAMNet + SVM model for audio classification.

---

### **Download & Extract Files**

```sh
unzip Distress_Binary_Classification_Model.pkl.zip -d model/
```

### **Using the Model for Predictions**

```python
import pickle
import numpy as np
from inference_script import preprocess_audio, predict

# Load the model
with open('model/Distress_Binary_Classification_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example usage with a 5-second audio chunk
audio_chunk = preprocess_audio("example.wav")
prediction = predict(model, audio_chunk)
print("Predicted class:", prediction)
```

### **Best Practices**
- Ensure audio input is **5 seconds long** for optimal predictions.
- Use `.wav` format with a sample rate of **16kHz** for best compatibility.
