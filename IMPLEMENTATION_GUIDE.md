# AI-Based PCB Defect Detection - Complete Implementation Guide
## Step-by-Step Python Project for ECE Students using VS Code

---

## 1. Project Overview

### What is this Project?

This project teaches you how to build an **AI system** that automatically detects defects (like short circuits, missing components, broken traces) in PCB images using:

- **Machine Learning**: Training algorithms to recognize patterns
- **Computer Vision**: Processing and analyzing images
- **Deep Learning (CNN)**: Convolutional Neural Networks for image classification

### How It Works

```
PCB Image Input
    ↓
Pre-process Image (resize, normalize)
    ↓
Feed to CNN Model (trained on defect patterns)
    ↓
Model outputs decision: "DEFECTIVE" or "NORMAL"
    ↓
Display Result with Confidence Score
```

### What You'll Learn

✅ Image processing with OpenCV  
✅ Building neural networks with TensorFlow  
✅ Training and testing ML models  
✅ Visualizing results with graphs  
✅ Python programming for AI/ML  

---

## 2. Required Software Setup

### Step 2.1: Install Python

**Download Python 3.9 or higher** from: https://www.python.org/downloads/

During installation:
- ✅ Check "Add Python to PATH"
- ✅ Check "Install pip"

**Verify installation:**
```bash
python --version
pip --version
```

### Step 2.2: Install VS Code

Download from: https://code.visualstudio.com/

Install extensions:
1. **Python** (by Microsoft)
2. **Pylance** (for code suggestions)
3. **Jupyter** (for notebooks)

### Step 2.3: Install Required Python Libraries

Create a file named `requirements.txt` in your project folder:

```
opencv-python==4.8.0
numpy==1.24.0
matplotlib==3.7.0
tensorflow==2.13.0
scikit-learn==1.3.0
pandas==2.0.0
Pillow==10.0.0
```

**Install all packages at once:**

```bash
pip install -r requirements.txt
```

**Or install individually:**

```bash
pip install opencv-python numpy matplotlib tensorflow scikit-learn pandas Pillow
```

**Verify Installation:**

```bash
python -c "import cv2, numpy, tensorflow; print('All libraries installed successfully!')"
```

---

## 3. Project Folder Structure

### Create Project Folder Structure

Open VS Code, create a new folder called `PCB_Defect_Detection_Project`:

```
PCB_Defect_Detection_Project/
│
├── dataset/
│   ├── train/
│   │   ├── defective/          (defective PCB images)
│   │   └── normal/             (normal PCB images)
│   └── test/
│       ├── defective/          (test defective images)
│       └── normal/             (test normal images)
│
├── models/
│   └── pcb_defect_model.h5     (trained model - created later)
│
├── output/
│   ├── accuracy_plot.png       (results visualization)
│   ├── confusion_matrix.png
│   └── predictions.csv
│
├── main.py                      (main project file)
├── train_model.py              (training script)
├── predict.py                  (prediction script)
├── preprocess.py               (data preprocessing)
└── requirements.txt            (library versions)
```

### Create Folders in VS Code Terminal

Open terminal in VS Code:

```bash
# Create main folders
mkdir dataset models output
mkdir dataset/train dataset/test
mkdir dataset/train/defective dataset/train/normal
mkdir dataset/test/defective dataset/test/normal
```

### Purpose of Each Folder

| Folder | Purpose |
|--------|---------|
| `dataset/train/` | Training images (70% of data) |
| `dataset/test/` | Testing images (30% of data) |
| `models/` | Saves trained model for later use |
| `output/` | Stores graphs and results |

---

## 4. Dataset Preparation

### Where to Get PCB Images

**Option 1: Kaggle (Recommended for Beginners)**

Download free PCB defect datasets:
- Search: "PCB Defect Detection" on kaggle.com
- Popular dataset: "PCB Defect Detection" (~300+ images)

**Option 2: Create Synthetic Dataset**

Use this Python code to generate synthetic images:

**File: `create_synthetic_dataset.py`**

```python
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw

def create_synthetic_pcb_images(output_dir, num_images=100):
    """
    Create synthetic PCB images for training
    num_images: total images to create (50% normal, 50% defective)
    """
    
    os.makedirs(f"{output_dir}/train/normal", exist_ok=True)
    os.makedirs(f"{output_dir}/train/defective", exist_ok=True)
    os.makedirs(f"{output_dir}/test/normal", exist_ok=True)
    os.makedirs(f"{output_dir}/test/defective", exist_ok=True)
    
    # Create NORMAL PCB images
    for i in range(num_images // 4):
        image = Image.new('RGB', (224, 224), color='green')
        draw = ImageDraw.Draw(image)
        
        # Draw circuit traces and components
        for _ in range(20):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            draw.line([(x1, y1), (x2, y2)], fill='yellow', width=2)
        
        # Draw components (rectangles)
        for _ in range(15):
            x1, y1 = np.random.randint(0, 200, 2)
            draw.rectangle([x1, y1, x1+20, y1+15], outline='white', width=2)
        
        image.save(f"{output_dir}/train/normal/normal_{i}.jpg")
    
    # Create DEFECTIVE PCB images (with visible defects)
    for i in range(num_images // 4):
        image = Image.new('RGB', (224, 224), color='green')
        draw = ImageDraw.Draw(image)
        
        # Draw normal circuit first
        for _ in range(15):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            draw.line([(x1, y1), (x2, y2)], fill='yellow', width=2)
        
        # Add defect: broken trace (red line)
        x1, y1 = np.random.randint(20, 200, 2)
        draw.line([(x1, y1), (x1+50, y1+50)], fill='red', width=3)
        
        # Add another defect: missing component (black circle)
        cx, cy = np.random.randint(50, 180, 2)
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill='black')
        
        image.save(f"{output_dir}/train/defective/defective_{i}.jpg")
    
    # Test set (split same way)
    for i in range(num_images // 4):
        image = Image.new('RGB', (224, 224), color='green')
        draw = ImageDraw.Draw(image)
        for _ in range(20):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            draw.line([(x1, y1), (x2, y2)], fill='yellow', width=2)
        for _ in range(15):
            x1, y1 = np.random.randint(0, 200, 2)
            draw.rectangle([x1, y1, x1+20, y1+15], outline='white', width=2)
        image.save(f"{output_dir}/test/normal/normal_test_{i}.jpg")
    
    for i in range(num_images // 4):
        image = Image.new('RGB', (224, 224), color='green')
        draw = ImageDraw.Draw(image)
        for _ in range(15):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            draw.line([(x1, y1), (x2, y2)], fill='yellow', width=2)
        x1, y1 = np.random.randint(20, 200, 2)
        draw.line([(x1, y1), (x1+50, y1+50)], fill='red', width=3)
        cx, cy = np.random.randint(50, 180, 2)
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill='black')
        image.save(f"{output_dir}/test/defective/defective_test_{i}.jpg")
    
    print(f"✅ Created synthetic dataset with {num_images} images")

# Run this to create synthetic dataset
if __name__ == "__main__":
    create_synthetic_pcb_images("dataset", num_images=200)
```

**Run in VS Code Terminal:**

```bash
python create_synthetic_dataset.py
```

### Dataset Structure After Download/Creation

Your dataset should look like:

```
dataset/
├── train/
│   ├── defective/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ... (50+ images)
│   └── normal/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ... (50+ images)
└── test/
    ├── defective/
    │   └── ... (20+ images)
    └── normal/
        └── ... (20+ images)
```

---

## 5. Image Preprocessing using OpenCV

### What is Preprocessing?

Converting raw images into a format that the AI model can understand:

```
Original Image (any size)
    ↓
Resize to 224×224
    ↓
Convert to RGB format
    ↓
Normalize values (0-1)
    ↓
Ready for AI model
```

### Preprocessing Code

**File: `preprocess.py`**

```python
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class ImagePreprocessor:
    """
    Handles image loading and preprocessing for PCB defect detection
    """
    
    def __init__(self, img_size=224):
        self.img_size = img_size  # Standard size for CNN input
    
    def load_image(self, image_path):
        """
        Load a single image from file path
        
        Args:
            image_path: path to image file
        
        Returns:
            preprocessed image as numpy array
        """
        
        # Read image from file
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Resize image to standard size
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values from [0, 255] to [0, 1]
        image = image.astype('float32') / 255.0
        
        return image
    
    def load_dataset(self, dataset_path):
        """
        Load entire dataset from folder structure
        Expected structure:
            dataset_path/
            ├── defective/
            │   ├── image1.jpg
            │   └── ...
            └── normal/
                ├── image1.jpg
                └── ...
        
        Returns:
            X: array of images
            y: array of labels (0=normal, 1=defective)
            filenames: list of original filenames
        """
        
        X = []  # Images
        y = []  # Labels
        filenames = []
        
        # Load NORMAL images (label = 0)
        normal_path = os.path.join(dataset_path, 'normal')
        if os.path.exists(normal_path):
            for filename in os.listdir(normal_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(normal_path, filename)
                    image = self.load_image(image_path)
                    
                    if image is not None:
                        X.append(image)
                        y.append(0)  # 0 = normal
                        filenames.append(filename)
        
        # Load DEFECTIVE images (label = 1)
        defective_path = os.path.join(dataset_path, 'defective')
        if os.path.exists(defective_path):
            for filename in os.listdir(defective_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(defective_path, filename)
                    image = self.load_image(image_path)
                    
                    if image is not None:
                        X.append(image)
                        y.append(1)  # 1 = defective
                        filenames.append(filename)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Loaded {len(X)} images")
        print(f"   Normal images: {np.sum(y == 0)}")
        print(f"   Defective images: {np.sum(y == 1)}")
        
        return X, y, filenames
    
    def apply_contrast_enhancement(self, image):
        """
        Enhance image contrast to highlight defects
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        
        # Convert to grayscale for histogram equalization
        gray = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def apply_augmentation(self, image):
        """
        Apply random transformations for training robustness
        Helps model generalize better
        """
        
        # Random rotation (±10 degrees)
        h, w = image.shape[:2]
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
        
        return image


# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = ImagePreprocessor(img_size=224)
    
    # Load dataset
    train_X, train_y, _ = preprocessor.load_dataset("dataset/train")
    test_X, test_y, _ = preprocessor.load_dataset("dataset/test")
    
    print(f"Training set shape: {train_X.shape}")
    print(f"Test set shape: {test_X.shape}")
```

---

## 6. Building the CNN Model

### What is a CNN?

A **Convolutional Neural Network (CNN)** is a deep learning model designed to process images:

```
Input Image (224×224×3)
    ↓
Convolutional Layer (learns basic features like edges)
    ↓
Pooling Layer (makes data smaller)
    ↓
Convolutional Layer (learns complex features)
    ↓
Pooling Layer
    ↓
Flatten Layer (convert 3D to 1D)
    ↓
Dense Layers (make final decision)
    ↓
Output (Probability: Defective or Normal)
```

### Building CNN Model Code

**File: `model.py`**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class PCBDefectCNN:
    """
    Convolutional Neural Network for PCB Defect Detection
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize CNN model architecture
        
        Args:
            input_shape: shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """
        Build CNN architecture
        """
        
        model = models.Sequential([
            
            # ===== BLOCK 1: Feature Detection =====
            # Convolution: Learn basic features (edges, corners)
            layers.Conv2D(
                filters=32,           # 32 different filters
                kernel_size=(3, 3),   # 3×3 filter size
                activation='relu',    # ReLU activation function
                padding='same',       # Keep same image size
                input_shape=self.input_shape
            ),
            # Reduce image size (224×224 → 112×112)
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # ===== BLOCK 2: More Complex Features =====
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # (112×112 → 56×56)
            
            # ===== BLOCK 3: Higher Level Features =====
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # (56×56 → 28×28)
            
            # ===== BLOCK 4: Deep Features =====
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # (28×28 → 14×14)
            
            # ===== FLATTEN: Convert 3D data to 1D =====
            layers.Flatten(),
            
            # ===== DENSE LAYERS: Classification =====
            # First dense layer with 256 neurons
            layers.Dense(256, activation='relu'),
            
            # Dropout: Randomly disable 50% of neurons during training
            # This prevents overfitting
            layers.Dropout(0.5),
            
            # Second dense layer with 128 neurons
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # ===== OUTPUT LAYER =====
            # 2 neurons: [probability_normal, probability_defective]
            # Softmax converts outputs to probabilities (0-1, sum=1)
            layers.Dense(2, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate: how fast the model learns
        """
        
        # Adam optimizer: adaptive learning rate
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Categorical crossentropy: standard loss for classification
        loss = 'categorical_crossentropy'
        
        # Metrics to monitor: accuracy
        metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("✅ Model compiled successfully")
    
    def get_model_summary(self):
        """
        Display model architecture
        """
        if self.model is None:
            print("❌ Model not built yet. Call build_model() first.")
            return
        
        self.model.summary()
        
        # Print total parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")


# Example: Create and display model
if __name__ == "__main__":
    # Create CNN
    cnn = PCBDefectCNN(input_shape=(224, 224, 3))
    
    # Build model
    cnn.build_model()
    
    # Compile model
    cnn.compile_model(learning_rate=0.001)
    
    # Show model architecture
    cnn.get_model_summary()
```

---

## 7. Training the Model

### What Happens During Training?

```
For each EPOCH (repetition):
  1. Load batch of training images
  2. Pass through model (Forward Pass)
  3. Calculate error (Loss)
  4. Calculate how to improve (Backpropagation)
  5. Update model weights
  6. Validate on validation set
  7. Check if performance improved
```

### Training Code

**File: `train_model.py`**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from preprocess import ImagePreprocessor
from model import PCBDefectCNN

class ModelTrainer:
    """
    Train the PCB defect detection model
    """
    
    def __init__(self, model_path="models/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_data(self, train_X, train_y, test_size=0.2):
        """
        Prepare training and validation data
        
        Args:
            train_X: training images
            train_y: training labels
            test_size: fraction for validation (0.2 = 20%)
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        
        # Split: 80% training, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_X, train_y,
            test_size=test_size,
            random_state=42,
            stratify=train_y  # Keep class balance
        )
        
        # Convert labels to one-hot encoding
        # [0] → [1, 0] (normal)
        # [1] → [0, 1] (defective)
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        
        print(f"✅ Data prepared:")
        print(f"   Training: {X_train.shape[0]} images")
        print(f"   Validation: {X_val.shape[0]} images")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, X_train, X_val, y_train, y_val, epochs=20, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: training images
            X_val: validation images
            y_train: training labels (one-hot encoded)
            y_val: validation labels
            epochs: number of training cycles
            batch_size: images per batch
        
        Returns:
            history: training history for plotting
        """
        
        # Create model
        cnn = PCBDefectCNN()
        cnn.build_model()
        cnn.compile_model(learning_rate=0.001)
        
        # Early stopping: stop if validation loss doesn't improve
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,           # Stop after 5 epochs without improvement
            restore_best_weights=True
        )
        
        # Model checkpoint: save best model
        checkpoint = keras.callbacks.ModelCheckpoint(
            f'{self.model_path}pcb_defect_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        print("\n🔄 Starting training...\n")
        
        # Train model
        history = cnn.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1  # Show progress
        )
        
        print("\n✅ Training completed!")
        
        return history, cnn.model
    
    def plot_training_history(self, history, save_path="output/"):
        """
        Plot training and validation accuracy/loss
        """
        
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Loss
        axes[1].plot(history.history['loss'], label='Training Loss', marker='o', color='red')
        axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s', color='orange')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}training_history.png', dpi=100)
        print(f"✅ Saved training plot to {save_path}training_history.png")
        plt.close()


# Main training script
if __name__ == "__main__":
    
    print("="*60)
    print("PCB DEFECT DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Step 1: Load dataset
    print("\n📂 Loading dataset...")
    preprocessor = ImagePreprocessor(img_size=224)
    X_train, y_train, _ = preprocessor.load_dataset("dataset/train")
    
    # Step 2: Prepare data
    print("\n📊 Preparing data...")
    trainer = ModelTrainer()
    X_tr, X_val, y_tr, y_val = trainer.prepare_data(X_train, y_train, test_size=0.2)
    
    # Step 3: Train model
    print("\n🤖 Training model...")
    history, model = trainer.train(
        X_tr, X_val, y_tr, y_val,
        epochs=20,
        batch_size=32
    )
    
    # Step 4: Plot results
    print("\n📈 Plotting results...")
    trainer.plot_training_history(history)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
```

**Run training in VS Code terminal:**

```bash
python train_model.py
```

---

## 8. Testing the Model on New Images

### What is Testing?

Using trained model to predict on images it has **never seen** before.

### Prediction Code

**File: `predict.py`**

```python
import tensorflow as tf
import numpy as np
from preprocess import ImagePreprocessor
import os
import json

class PCBDefectPredictor:
    """
    Make predictions using trained model
    """
    
    def __init__(self, model_path="models/pcb_defect_model.h5"):
        """
        Load trained model
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Normal', 'Defective']
        self.preprocessor = ImagePreprocessor(img_size=224)
        print(f"✅ Model loaded from {model_path}")
    
    def predict_single_image(self, image_path, confidence_threshold=0.5):
        """
        Predict defect for a single PCB image
        
        Args:
            image_path: path to PCB image
            confidence_threshold: minimum confidence to classify as defective
        
        Returns:
            prediction dict with result and confidence
        """
        
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        
        if image is None:
            return {'error': 'Could not load image'}
        
        # Prepare for model input (add batch dimension)
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Get probabilities
        prob_normal, prob_defective = predictions[0]
        
        # Determine class
        if prob_defective >= confidence_threshold:
            predicted_class = 'DEFECTIVE'
        else:
            predicted_class = 'NORMAL'
        
        confidence = max(prob_normal, prob_defective) * 100
        
        result = {
            'image_name': os.path.basename(image_path),
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'probability_normal': f"{prob_normal*100:.2f}%",
            'probability_defective': f"{prob_defective*100:.2f}%"
        }
        
        return result
    
    def predict_batch(self, test_image_dir):
        """
        Predict on all images in a folder
        
        Args:
            test_image_dir: folder containing test images
        
        Returns:
            list of predictions
        """
        
        predictions = []
        image_files = [f for f in os.listdir(test_image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n🔍 Testing {len(image_files)} images...\n")
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(test_image_dir, filename)
            result = self.predict_single_image(image_path)
            predictions.append(result)
            
            # Display result
            print(f"{i}. {result['image_name']}")
            print(f"   → Prediction: {result['prediction']}")
            print(f"   → Confidence: {result['confidence']}")
            print()
        
        return predictions
    
    def save_predictions_csv(self, predictions, output_file="output/predictions.csv"):
        """
        Save predictions to CSV file
        """
        
        import csv
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
        
        print(f"✅ Predictions saved to {output_file}")


# Main prediction script
if __name__ == "__main__":
    
    print("="*60)
    print("PCB DEFECT DETECTION - MODEL PREDICTION")
    print("="*60)
    
    # Load model
    print("\n🤖 Loading trained model...")
    predictor = PCBDefectPredictor("models/pcb_defect_model.h5")
    
    # Test on single image
    print("\n📸 Testing single image...")
    result = predictor.predict_single_image("dataset/test/normal/normal_test_0.jpg")
    print(result)
    
    # Test on all test images
    print("\n📸 Testing all images in test folder...")
    predictions = predictor.predict_batch("dataset/test")
    
    # Save results
    predictor.save_predictions_csv(predictions)
    
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)
```

**Run predictions in VS Code terminal:**

```bash
python predict.py
```

---

## 9. Result Visualization

### Visualize Model Performance

**File: `visualize_results.py`**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tensorflow.keras.utils import to_categorical

from preprocess import ImagePreprocessor

class ResultVisualizer:
    """
    Visualize model results and performance metrics
    """
    
    def __init__(self, model_path="models/pcb_defect_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = ImagePreprocessor(img_size=224)
    
    def evaluate_on_test_set(self, test_X, test_y):
        """
        Evaluate model on test set and return metrics
        """
        
        # Convert labels to one-hot
        test_y_encoded = to_categorical(test_y, num_classes=2)
        
        # Get predictions
        predictions = self.model.predict(test_X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == test_y)
        
        print(f"✅ Test Set Accuracy: {accuracy*100:.2f}%")
        
        return predicted_classes, predictions, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path="output/"):
        """
        Plot confusion matrix heatmap
        """
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Defective'],
                   yticklabels=['Normal', 'Defective'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}confusion_matrix.png', dpi=100)
        print(f"✅ Saved confusion matrix to {save_path}confusion_matrix.png")
        plt.close()
    
    def plot_predictions_samples(self, test_X, test_y, num_samples=12, save_path="output/"):
        """
        Display sample predictions with images
        """
        
        predictions = self.model.predict(test_X[:num_samples], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(test_X[i])
            
            true_label = 'Normal' if test_y[i] == 0 else 'Defective'
            pred_label = 'Normal' if predicted_classes[i] == 0 else 'Defective'
            confidence = np.max(predictions[i]) * 100
            
            # Color: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            axes[i].set_title(title, color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}sample_predictions.png', dpi=100)
        print(f"✅ Saved sample predictions to {save_path}sample_predictions.png")
        plt.close()
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification metrics
        """
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Defective']
        )
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)


# Main visualization script
if __name__ == "__main__":
    
    print("="*60)
    print("PCB DEFECT DETECTION - RESULT VISUALIZATION")
    print("="*60)
    
    # Load test data
    print("\n📂 Loading test dataset...")
    preprocessor = ImagePreprocessor(img_size=224)
    test_X, test_y, _ = preprocessor.load_dataset("dataset/test")
    
    # Evaluate and visualize
    print("\n📊 Evaluating model...")
    visualizer = ResultVisualizer("models/pcb_defect_model.h5")
    
    predicted_classes, predictions, accuracy = visualizer.evaluate_on_test_set(test_X, test_y)
    
    print("\n📈 Creating visualizations...")
    visualizer.plot_confusion_matrix(test_y, predicted_classes)
    visualizer.plot_predictions_samples(test_X, test_y, num_samples=12)
    visualizer.print_classification_report(test_y, predicted_classes)
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*60)
```

**Run visualization in VS Code terminal:**

```bash
python visualize_results.py
```

---

## 10. Running the Project in VS Code

### Step-by-Step Execution

#### **Step 1: Create Project Folder**

```bash
mkdir PCB_Defect_Detection_Project
cd PCB_Defect_Detection_Project
```

#### **Step 2: Open in VS Code**

```bash
code .
```

#### **Step 3: Create Project Structure**

```bash
mkdir dataset models output
mkdir dataset/train dataset/test
mkdir dataset/train/defective dataset/train/normal
mkdir dataset/test/defective dataset/test/normal
```

#### **Step 4: Create requirements.txt**

Create file: `requirements.txt`

```
opencv-python==4.8.0
numpy==1.24.0
matplotlib==3.7.0
tensorflow==2.13.0
scikit-learn==1.3.0
pandas==2.0.0
Pillow==10.0.0
seaborn==0.12.0
```

#### **Step 5: Install Dependencies**

In VS Code terminal:

```bash
pip install -r requirements.txt
```

#### **Step 6: Create Dataset**

Copy your PCB images to:
- `dataset/train/normal/` (normal PCBs)
- `dataset/train/defective/` (defective PCBs)
- `dataset/test/normal/` (test normal)
- `dataset/test/defective/` (test defective)

Or generate synthetic dataset:

```bash
python create_synthetic_dataset.py
```

#### **Step 7: Train Model**

```bash
python train_model.py
```

**This will:**
- Load images from `dataset/train/`
- Train CNN model for 20 epochs
- Save model to `models/pcb_defect_model.h5`
- Create `output/training_history.png` with graphs

#### **Step 8: Test Model**

```bash
python predict.py
```

**This will:**
- Load trained model
- Test on all images in `dataset/test/`
- Save predictions to `output/predictions.csv`

#### **Step 9: Visualize Results**

```bash
python visualize_results.py
```

**This will:**
- Create confusion matrix heatmap
- Show sample predictions with images
- Print detailed metrics

#### **Optional: Combined Training Script**

**File: `main.py`** (Run everything at once)

```python
import os
os.system('python create_synthetic_dataset.py')
os.system('python train_model.py')
os.system('python visualize_results.py')
os.system('python predict.py')
```

Run with:

```bash
python main.py
```

---

## 11. Expected Output

### Console Output Examples

**After training:**

```
═══════════════════════════════════════════════════════
PCB DEFECT DETECTION - MODEL TRAINING
═══════════════════════════════════════════════════════

📂 Loading dataset...
✅ Loaded 150 images
   Normal images: 75
   Defective images: 75

📊 Preparing data...
✅ Data prepared:
   Training: 120 images
   Validation: 30 images

🤖 Training model...
Epoch 1/20
4/4 [==============================] - 3s 8ms/step - loss: 0.6923 - accuracy: 0.5250
- val_loss: 0.6890 - val_accuracy: 0.5667

Epoch 2/20
4/4 [==============================] - 1s 8ms/step - loss: 0.6511 - accuracy: 0.7583
- val_loss: 0.5892 - val_accuracy: 0.8333

...

Epoch 20/20
4/4 [==============================] - 1s 8ms/step - loss: 0.0234 - accuracy: 0.9917
- val_loss: 0.0445 - val_accuracy: 0.9667

✅ Training completed!

📈 Plotting results...
✅ Saved training plot to output/training_history.png

═══════════════════════════════════════════════════════
✅ TRAINING COMPLETE!
═══════════════════════════════════════════════════════
```

**After prediction:**

```
═══════════════════════════════════════════════════════
PCB DEFECT DETECTION - MODEL PREDICTION
═══════════════════════════════════════════════════════

🤖 Loading trained model...
✅ Model loaded from models/pcb_defect_model.h5

📸 Testing single image...
{'image_name': 'normal_test_0.jpg', 'prediction': 'NORMAL', 
'confidence': '96.45%', 'probability_normal': '96.45%', 
'probability_defective': '3.55%'}

📸 Testing all images in test folder...

🔍 Testing 30 images...

1. normal_test_0.jpg
   → Prediction: NORMAL
   → Confidence: 96.45%

2. defective_test_0.jpg
   → Prediction: DEFECTIVE
   → Confidence: 94.23%

...

✅ Predictions saved to output/predictions.csv

═══════════════════════════════════════════════════════
✅ PREDICTION COMPLETE!
═══════════════════════════════════════════════════════
```

### Generated Files

After running all scripts, you'll have:

```
PCB_Defect_Detection_Project/
├── models/
│   └── pcb_defect_model.h5          ✅ Trained model
├── output/
│   ├── training_history.png         ✅ Accuracy/Loss graphs
│   ├── confusion_matrix.png         ✅ Confusion matrix heatmap
│   ├── sample_predictions.png       ✅ Sample test images with predictions
│   └── predictions.csv              ✅ All predictions in table format
└── dataset/
    ├── train/
    │   ├── normal/
    │   └── defective/
    └── test/
        ├── normal/
        └── defective/
```

---

## 12. Improvements for Future

### Easy Enhancements

#### **1. Real-Time Camera Inspection**

**Add webcam detection:**

```python
import cv2

cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    
    # Resize and preprocess
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    
    # Predict
    prediction = model.predict(np.expand_dims(frame_normalized, 0))
    label = 'DEFECTIVE' if prediction[0][1] > 0.5 else 'NORMAL'
    
    # Display
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('PCB Inspection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### **2. Use Larger Dataset**

- Download from Kaggle (500+ images)
- Improves accuracy significantly
- Better generalization

#### **3. Transfer Learning (Faster Training)**

Use pre-trained models:

```python
from tensorflow.keras.applications import ResNet50

# Use pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
```

#### **4. Deploy as Web App (Flask)**

```python
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/pcb_defect_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # Process image and predict
    result = {'prediction': 'DEFECTIVE', 'confidence': '95%'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

#### **5. YOLOv8 for Object Detection**

Detect AND locate defects:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict('pcb_image.jpg')
```

#### **6. Explainability (LIME / SHAP)**

Explain which parts of image matter:

```python
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
exp = explainer.explain_instance(image, model.predict, top_labels=2)
```

---

## Summary: Complete Step-by-Step

| Step | Command | Output |
|------|---------|--------|
| 1. Create folder | `mkdir PCB_Defect_Detection_Project` | Folder created |
| 2. Install packages | `pip install -r requirements.txt` | Libraries installed |
| 3. Create dataset | `python create_synthetic_dataset.py` | 200 images created |
| 4. Train model | `python train_model.py` | Model saved, graphs created |
| 5. Test model | `python predict.py` | Predictions & CSV |
| 6. Visualize | `python visualize_results.py` | Confusion matrix, metrics |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| `No images found` | Copy PCB images to `dataset/train/normal/` etc |
| `GPU not found` | Works fine on CPU (just slower) |
| `Model not found` | Make sure `train_model.py` completed successfully |
| `Out of memory` | Reduce batch_size from 32 to 16 in train_model.py |

---

**Congratulations!** You now have a complete AI-ML project for PCB defect detection! 🎉
