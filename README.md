# PCB Defect Detection using Deep Learning & Computer Vision
## Complete Implementation Guide for Kaggle Dataset

---

## 📋 Project Overview

This project uses **Convolutional Neural Networks (CNN)** to automatically classify PCB (Printed Circuit Board) defects into 6 categories using Kaggle datasets:

- **missing_hole** - PCB holes that should exist but are missing
- **mouse_bite** - Material bites along PCB traces
- **open_circuit** - Breaks in circuit traces
- **short** - Unintended connections between traces
- **spur** - Unwanted protruding traces
- **spurious_copper** - Unexpected copper deposits

**Key Features:**
- ✅ Deep CNN architecture with data augmentation
- ✅ Batch normalization and dropout for better generalization
- ✅ Multi-class classification (6 defect types)
- ✅ Complete training and prediction pipeline
- ✅ Result visualization and CSV export
- ✅ Command-line interface for easy usage

---

## 🛠️ System Requirements

### Minimum Requirements:
- Python 3.8+
- 4GB RAM (8GB recommended)
- GPU optional (CPU works, but slower)

### Recommended System:
- Python 3.9 or 3.10
- 8GB+ RAM
- NVIDIA GPU with CUDA support (for faster training)

---

## 📦 Installation & Setup

### Step 1: Create Project Folder

```bash
# Create and navigate to project folder
mkdir PCB_Defect_Detection
cd PCB_Defect_Detection

# Open in VS Code
code .
```

### Step 2: Download Dataset from Kaggle

**Option A: Using Kaggle Website (Recommended for Beginners)**

1. Go to: https://www.kaggle.com/
2. Search for: "PCB Defect Detection"
3. Find dataset by "Tangling" or similar
4. Download the dataset
5. Extract to project folder as `dataset/`

**Your folder structure should look like:**

```
PCB_Defect_Detection/
│
├── dataset/
│   ├── missing_hole/         (contains .jpg files)
│   ├── mouse_bite/           (contains .jpg files)
│   ├── open_circuit/         (contains .jpg files)
│   ├── short/                (contains .jpg files)
│   ├── spur/                 (contains .jpg files)
│   └── spurious_copper/      (contains .jpg files)
│
├── train_model.py            (training script)
├── predict_pcb.py            (prediction script)
└── requirements.txt          (dependencies)
```

**Option B: Using Kaggle CLI (Advanced)**

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (download from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d <dataset-id>

# Extract
unzip <dataset-name>.zip
```

### Step 3: Create Virtual Environment (Optional but Recommended)

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **TensorFlow 2.13**: Deep learning framework
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data handling

**Verify Installation:**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

---

## 🚀 Usage Guide

### Option 1: Train Model (Full Pipeline)

```bash
python train_model.py
```

**What happens:**
1. ✅ Loads all images from `dataset/` folders
2. ✅ Splits data: 64% training, 16% validation, 20% test
3. ✅ Builds CNN model (4 conv blocks, dropout, batch norm)
4. ✅ Trains model for ~20 epochs with early stopping
5. ✅ Saves best model to `models/pcb_defect_model.h5`
6. ✅ Saves training plots to `output/training_history.png`

**Expected Output:**

```
════════════════════════════════════════════════════════════════
█  PCB DEFECT DETECTION - MODEL TRAINING PIPELINE        █
════════════════════════════════════════════════════════════════

📂 STEP 1: LOADING DATASET
═════════════════════════════════════════════════════════════════
✅ Found 6 defect classes:
   1. missing_hole
   2. mouse_bite
   3. open_circuit
   4. short
   5. spur
   6. spurious_copper

   Loading missing_hole: 143 images...
   Loading mouse_bite: 152 images...
   ...
   Total images loaded: 915

📊 Class distribution:
   missing_hole: 143 images (15.6%)
   mouse_bite: 152 images (16.6%)
   ...

🚀 STEP 5: TRAINING MODEL
────────────────────────────
Epoch 1/20
15/15 [==============================] - 12s 800ms/step - loss: 1.7823 - accuracy: 0.2944 - val_loss: 1.6234 - val_accuracy: 0.3889

Epoch 2/20
15/15 [==============================] - 8s 530ms/step - loss: 1.3456 - accuracy: 0.5233 - val_loss: 1.1123 - val_accuracy: 0.6111
...

✅ TRAINING PIPELINE COMPLETED!
────────────────────────────────
📊 Training Summary:
   Total Classes: 6
   Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
   Final Test Accuracy: 94.23%
   Model Saved: models/pcb_defect_model.h5
   Plots Saved: output/training_history.png
```

**Training Time:**
- CPU: 30-60 minutes (depends on dataset size)
- GPU: 5-15 minutes

---

### Option 2: Predict on Single Image

```bash
python predict_pcb.py --image dataset/missing_hole/image_001.jpg --visualize
```

**Output:**

```
════════════════════════════════════════════════════════════════
█  PCB DEFECT DETECTION - PREDICTION & CLASSIFICATION PIPELINE
════════════════════════════════════════════════════════════════

🤖 INITIALIZING PREDICTOR
════════════════════════════════════════════════════════════════
🤖 Loading model from models/pcb_defect_model.h5...
✅ Model loaded successfully

📸 PREDICTING SINGLE IMAGE
════════════════════════════════════════════════════════════════
Image: dataset/missing_hole/image_001.jpg

✅ Prediction: missing_hole
   Confidence: 96.32%

   All probabilities:
   • missing_hole: 96.32%
   • mouse_bite: 2.15%
   • open_circuit: 0.89%
   • short: 0.45%
   • spur: 0.12%
   • spurious_copper: 0.07%
```

---

### Option 3: Batch Predict (Folder of Images)

```bash
python predict_pcb.py --folder dataset/test/ --save-csv --visualize
```

**Outputs:**
- Console output with 30+ predictions
- `output/predictions.csv` with detailed results
- Grid visualization of predictions

**CSV Format Example:**

```csv
image_name,predicted_class,confidence,prob_missing_hole,prob_mouse_bite,...
image_001.jpg,missing_hole,96.32%,96.32%,2.15%,...
image_002.jpg,open_circuit,91.45%,1.23%,0.56%,...
```

---

### Option 4: Default Prediction (Test Set)

```bash
python predict_pcb.py
```

Automatically predicts on all images in `dataset/test/` folder and saves results.

---

## 📊 Understanding the Model

### CNN Architecture

```
Input Image (224×224×3 RGB)
    ↓
Block 1: Conv(32) → Conv(32) → MaxPool(2x2) → BatchNorm
    ↓ (112×112)
Block 2: Conv(64) → Conv(64) → MaxPool(2x2) → BatchNorm
    ↓ (56×56)
Block 3: Conv(128) → Conv(128) → MaxPool(2x2) → BatchNorm
    ↓ (28×28)
Block 4: Conv(256) → Conv(256) → MaxPool(2x2) → BatchNorm
    ↓ (14×14)
Flatten
    ↓
Dense(512) → Dropout(0.5) → BatchNorm
    ↓
Dense(256) → Dropout(0.5) → BatchNorm
    ↓
Dense(6) → Softmax
    ↓
Output: Probabilities for 6 defect classes
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| **Conv2D** | Extract visual features (edges, patterns) |
| **MaxPooling** | Reduce dimensions, keep important features |
| **BatchNormalization** | Normalize layer outputs, faster convergence |
| **Dropout** | Prevent overfitting (randomly disable neurons) |
| **Dense Layers** | Make classification decision |
| **Softmax** | Convert to probabilities (sum = 1) |

### Hyperparameters

```python
IMG_SIZE = 224              # Standard CNN input size
BATCH_SIZE = 32             # Images per batch
EPOCHS = 20                 # Training cycles (auto-stops with early stopping)
LEARNING_RATE = 0.001       # How fast model learns
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for classification
```

---

## 📈 Interpreting Results

### Training Plots

After training, check `output/training_history.png`:

**Good Training Curve:**
```
Accuracy (should increase, then plateau)
├─ Training accuracy: target 95%+
└─ Validation accuracy: target 85%+

Loss (should decrease continuously)
├─ Training loss: target <0.1
└─ Validation loss: target <0.5
```

**Signs of Problems:**
- ❌ Accuracy stuck at random ~16.67% (1/6 classes) → Too early to judge
- ❌ Validation loss increasing while training loss decreases → Overfitting
- ❌ Both accuracies increasing but very slowly → Increase learning rate

### Prediction Confidence

**Interpretation:**

| Confidence | Meaning |
|-----------|---------|
| **>90%** | ✅ Very confident, reliable prediction |
| **70-90%** | ⚠️ Reasonably confident, probably correct |
| **50-70%** | ⚠️ Uncertain, may need manual review |
| **<50%** | ❌ Very uncertain, unreliable |

---

## 🔧 Troubleshooting

### Problem 1: "No module named 'tensorflow'"

**Solution:**
```bash
pip install --upgrade tensorflow
```

If still issues:
```bash
pip uninstall tensorflow -y
pip install tensorflow==2.13.0
```

### Problem 2: "Dataset folder not found"

**Solution:**
```bash
# Check correct path
ls dataset/
# or on Windows:
dir dataset\

# Make sure images are in subfolders:
dataset/missing_hole/image_001.jpg  (✅ correct)
dataset/image_001.jpg               (❌ wrong)
```

### Problem 3: "CUDA not found / GPU not detected"

**Solution:** 
- Don't worry, CPU works fine (just slower)
- Code automatically falls back to CPU
- For GPU, follow: https://www.tensorflow.org/install/pip#windows-native_1

### Problem 4: "Memory error" / "Out of memory"

**Solution:**
```python
# Edit train_model.py, change line:
BATCH_SIZE = 16  # Reduce from 32 to 16
```

### Problem 5: "Model not found" when predicting

**Solution:**
```bash
# Make sure training completed successfully
python train_model.py

# Check file exists:
ls models/pcb_defect_model.h5  # macOS/Linux
dir models\pcb_defect_model.h5 # Windows
```

---

## 📁 File Structure After Running

```
PCB_Defect_Detection/
│
├── dataset/
│   ├── missing_hole/
│   ├── mouse_bite/
│   ├── open_circuit/
│   ├── short/
│   ├── spur/
│   └── spurious_copper/
│
├── models/
│   └── pcb_defect_model.h5          ✅ Trained model (~50-100MB)
│
├── output/
│   ├── training_history.png         ✅ Accuracy/Loss graphs
│   └── predictions.csv              ✅ Batch prediction results
│
├── train_model.py                   (training script)
├── predict_pcb.py                   (prediction script)
└── requirements.txt                 (dependencies)
```

---

## 🚀 Advanced Usage

### Use GPU for Faster Training

```python
# TensorFlow automatically uses GPU if available
# Check GPU:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If CUDA-compatible GPU detected, it will be used automatically
# Training speed: ~2-3x faster with GPU
```

### Increase Model Accuracy

**Option 1: Use Larger Dataset**
- Minimum 300+ images recommended
- More images = better accuracy

**Option 2: Train Longer**
```python
EPOCHS = 50  # Instead of 20
```

**Option 3: Transfer Learning (Pre-trained Model)**
```python
# Use ResNet50 pre-trained on ImageNet
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False)
# Add custom layers on top
```

### Deploy as Web Application

```python
# Flask app for predictions via HTTP
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/pcb_defect_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # Process and predict...
    return jsonify({'prediction': 'short', 'confidence': '94.32%'})

if __name__ == '__main__':
    app.run()
```

---

## 📚 Learning Resources

**Understanding CNNs:**
- Visualization: https://www.youtube.com/watch?v=YRhxdVk_sIs (3Blue1Brown)
- Interactive: https://playground.tensorflow.org/

**TensorFlow & Keras:**
- Official Tutorial: https://www.tensorflow.org/tutorials/images/cnn
- Keras Documentation: https://keras.io/

**Computer Vision:**
- OpenCV Tutorials: https://docs.opencv.org/master/

---

## 📊 Performance Metrics

### Expected Accuracy

| Model Size | Training Data | Accuracy |
|--------|----|----------|
| Small | <300 images | 70-80% |
| Medium | 300-500 images | 80-90% |
| Large | 500+ images | 90-95%+ |

### Training Time (Reference)
- **Small Dataset (300 img)**: 5-10 min (GPU), 30-40 min (CPU)
- **Medium Dataset (500 img)**: 10-20 min (GPU), 60-90 min (CPU)
- **Large Dataset (1000+ img)**: 30-60 min (GPU), 3-5 hrs (CPU)

---

## ✅ Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] Kaggle dataset downloaded and extracted as `dataset/`
- [ ] Dataset has 6 folders: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- [ ] `requirements.txt` installed (`pip install -r requirements.txt`)
- [ ] `train_model.py` and `predict_pcb.py` in project root
- [ ] Folders created: `models/`, `output/`

---

## 💡 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python train_model.py

# 3. Predict single image
python predict_pcb.py --image dataset/missing_hole/image_001.jpg --visualize

# 4. Predict batch
python predict_pcb.py --folder dataset/test/ --save-csv --visualize

# 5. Default test
python predict_pcb.py
```

---

## 📞 Support & Questions

**Common Issues:**
1. Check error messages carefully
2. Look in Troubleshooting section
3. Verify dataset format

**Project Files:**
- `train_model.py`: Main training pipeline (well-commented)
- `predict_pcb.py`: Prediction interface (with CLI options)
- `requirements.txt`: All dependencies

---

**Project Version:** 1.0  
**Last Updated:** March 2024  
**Developed for:** Engineering Students & AI enthusiasts

---

## 🎓 What You'll Learn

✅ How to build CNN architectures  
✅ Image preprocessing with OpenCV  
✅ Training deep learning models  
✅ Making predictions on new data  
✅ Evaluating ML model performance  
✅ Visualizing training results  
✅ Handling real-world datasets  

Good luck with your project! 🚀
