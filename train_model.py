"""
PCB DEFECT DETECTION - MODEL TRAINING SCRIPT
============================================
This script trains a CNN model on Kaggle PCB defect dataset
to classify PCB images into defect categories.

Defect Classes:
- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

Author: AI-ML Project
Date: 2024
"""

import os
import argparse
import json
# pyright: reportMissingModuleSource=false
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import seaborn as sns

keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# ============================================================
# STEP 1: CONFIGURATION AND SETUP
# ============================================================

# Dataset configuration
DATASET_PATH = "PCB_DATASET/images"  # Path to attached dataset folder
IMG_SIZE = 224           # Image resize dimension
BATCH_SIZE = 32          # Training batch size
EPOCHS = 20              # Number of training epochs
LEARNING_RATE = 0.0003   # Model learning rate
MODEL_SAVE_PATH = "models/pcb_defect_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
MODEL_META_PATH = "models/model_meta.json"
HISTORY_PLOT_PATH = "output/training_history.png"

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ============================================================
# STEP 2: DATA LOADING AND PREPROCESSING
# ============================================================


class DataLoader:
    """
    Loads and preprocesses PCB images from Kaggle dataset
    """

    def __init__(self, dataset_path, img_size=224):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_names = []
        self.label_encoder = LabelEncoder()

    @staticmethod
    def _is_image_file(filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

    @staticmethod
    def _normalize_class_name(class_name):
        normalized = class_name.strip().lower().replace(" ", "_").replace("-", "_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def _discover_images_by_class(self):
        """
        Discover images and class labels from real-world uploaded dataset layouts.

        Supported layouts:
        1) dataset/class_name/image.jpg
        2) dataset/train/class_name/image.jpg (also val/valid/validation/test)
        3) Nested variants where split folders appear deeper in the tree
        """
        split_names = {"train", "val", "valid", "validation", "test"}
        images_by_class = {}

        for root, _, files in os.walk(self.dataset_path):
            image_files = [f for f in files if self._is_image_file(f)]
            if not image_files:
                continue

            root_path = os.path.normpath(root)
            rel_parts = os.path.relpath(
                root_path, self.dataset_path).split(os.sep)
            rel_parts = [p for p in rel_parts if p and p != "."]

            class_name = None

            # If split folder exists in the path, class is usually the next segment.
            for idx, part in enumerate(rel_parts):
                if part.lower() in split_names and idx + 1 < len(rel_parts):
                    class_name = rel_parts[idx + 1]
                    break

            # Fallback: immediate parent folder name.
            if class_name is None:
                class_name = os.path.basename(root_path)

            class_name = self._normalize_class_name(class_name)

            # Skip images that are directly under dataset root without class folder.
            if class_name.lower() in split_names or class_name == os.path.basename(self.dataset_path):
                continue

            images_by_class.setdefault(class_name, [])
            for image_file in image_files:
                images_by_class[class_name].append(
                    os.path.join(root_path, image_file))

        return images_by_class

    def load_images_from_folder(self):
        """
        Load all images from dataset folder structure

        Expected structure:
        dataset/
        ├── missing_hole/
        ├── mouse_bite/
        ├── open_circuit/
        ├── short/
        ├── spur/
        └── spurious_copper/

        Returns:
            X: numpy array of images (N, 224, 224, 3)
            y: numpy array of labels (N,)
            class_names: list of defect class names
        """

        images = []
        labels = []

        print("="*60)
        print("📂 LOADING DATASET FROM KAGGLE FORMAT")
        print("="*60)

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}. "
                "Create the folder or pass --dataset-path to point to your uploaded dataset."
            )

        images_by_class = self._discover_images_by_class()
        class_folders = sorted(
            [c for c, files in images_by_class.items() if len(files) > 0])

        if len(class_folders) < 2:
            raise ValueError(
                "Need at least 2 classes with images for training. "
                f"Found classes: {class_folders if class_folders else 'None'}"
            )

        self.class_names = class_folders

        print(f"\n✅ Found {len(class_folders)} defect classes:")
        for i, class_name in enumerate(class_folders):
            print(f"   {i+1}. {class_name}")

        # Load images for each class
        total_images = 0
        for class_idx, class_name in enumerate(class_folders):
            image_files = images_by_class.get(class_name, [])

            print(f"\n   Loading {class_name}: {len(image_files)} images...")

            for image_path in image_files:
                try:
                    # Read image
                    image = cv2.imread(image_path)

                    if image is None:
                        continue

                    # Resize to standard size
                    image = cv2.resize(image, (self.img_size, self.img_size))

                    # Convert BGR to RGB (OpenCV uses BGR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Normalize pixel values to [0, 1]
                    image = image.astype('float32') / 255.0

                    images.append(image)
                    labels.append(class_idx)
                    total_images += 1

                except Exception as e:
                    print(f"      ⚠️  Skipped {image_path}: {str(e)}")
                    continue

        print(f"\n✅ Total images loaded: {total_images}")

        if len(images) == 0:
            raise ValueError(
                "No readable images found in dataset. "
                "Check file extensions and dataset structure."
            )

        X = np.array(images)
        y = np.array(labels)

        print(f"\n📊 Dataset shape:")
        print(f"   Images: {X.shape}")
        print(f"   Labels: {y.shape}")

        # Print class distribution
        print(f"\n📈 Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for class_idx, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            print(
                f"   {self.class_names[class_idx]}: {count} images ({percentage:.1f}%)")

        return X, y

    def split_dataset(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split dataset into training, validation, and test sets

        Args:
            X: image array
            y: label array
            test_size: fraction for test set
            val_size: fraction of training set to use for validation

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """

        # First split: training vs test (80-20)
        # If classes are too imbalanced/small for stratification, fall back safely.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=None
            )

        # Second split: training vs validation (80-20 of training)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size,
                random_state=42,
                stratify=y_train
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size,
                random_state=42,
                stratify=None
            )

        print(f"\n✅ Dataset split:")
        print(f"   Training: {X_train.shape[0]} images")
        print(f"   Validation: {X_val.shape[0]} images")
        print(f"   Test: {X_test.shape[0]} images")

        return X_train, X_val, X_test, y_train, y_val, y_test


def save_class_names(class_names, class_names_path=CLASS_NAMES_PATH):
    """Save class names used in training for consistent prediction."""
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Saved class names to {class_names_path}")


def save_model_metadata(img_size, model_meta_path=MODEL_META_PATH):
    """Save model metadata required by prediction script."""
    with open(model_meta_path, "w", encoding="utf-8") as f:
        json.dump({"img_size": img_size}, f, indent=2)
    print(f"✅ Saved model metadata to {model_meta_path}")


# ============================================================
# STEP 3: BUILD CNN MODEL ARCHITECTURE
# ============================================================

def build_cnn_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Build Convolutional Neural Network (CNN) for PCB defect classification

    Architecture:
    - Input: 224×224×3 RGB image
    - Conv blocks with pooling
    - Flatten and dense layers
    - Output: Softmax with num_classes units

    Args:
        num_classes: number of defect classes
        input_shape: input image shape

    Returns:
        compiled keras model
    """

    print("\n" + "="*60)
    print("🏗️  BUILDING CNN MODEL")
    print("="*60)

    # Transfer learning backbone gives much stronger features than training
    # a deep CNN from scratch on this dataset size.
    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        print("✅ Loaded MobileNetV2 imagenet weights")
    except Exception as e:
        print(f"⚠️  Could not load imagenet weights ({e}), using random init")
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name='input_image')
    # Convert [0,1] input range from dataloader to MobileNetV2 expected [-1,1].
    x = layers.Rescaling(scale=2.0, offset=-1.0,
                         name='mobilenet_rescale')(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.35, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='pcb_mobilenetv2_classifier')

    # Print model architecture
    print("\n✅ Model architecture:")
    model.summary()

    return model


# ============================================================
# STEP 4: COMPILE MODEL
# ============================================================

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile model with optimizer, loss function, and metrics

    Args:
        model: keras model
        learning_rate: optimizer learning rate
    """

    # Adam optimizer: adaptive learning rate optimization
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Categorical crossentropy: standard loss for multi-class classification
    # L = -Σ(y_true * log(y_pred))
    loss = 'categorical_crossentropy'

    # Metrics to monitor
    metrics = ['accuracy']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    print("\n✅ Model compiled successfully")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Categorical Crossentropy")
    print(f"   Metrics: Accuracy")


# ============================================================
# STEP 5: DATA AUGMENTATION
# ============================================================

def get_data_augmentation():
    """
    Define data augmentation for training set
    Helps model generalize better by providing varied data
    """

    augmentation = ImageDataGenerator(
        rotation_range=20,        # Random rotation ±20°
        width_shift_range=0.2,    # Random horizontal shift 20%
        height_shift_range=0.2,   # Random vertical shift 20%
        horizontal_flip=True,     # Random horizontal flip
        zoom_range=0.2,           # Random zoom 0.8-1.2×
        brightness_range=[0.8, 1.2],  # Random brightness adjustment
        fill_mode='nearest'
    )

    return augmentation


# ============================================================
# STEP 6: TRAINING
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train the CNN model

    Args:
        model: compiled keras model
        X_train: training images
        y_train: training labels (one-hot encoded)
        X_val: validation images
        y_val: validation labels
        epochs: number of training cycles
        batch_size: images per batch

    Returns:
        history: training history object
    """

    print("\n" + "="*60)
    print("🚀 TRAINING MODEL")
    print("="*60)

    # Convert labels to one-hot encoding
    # [0] → [1, 0, 0, 0, 0, 0]
    # [1] → [0, 1, 0, 0, 0, 0]
    # etc.
    y_train_encoded = keras.utils.to_categorical(y_train)
    y_val_encoded = keras.utils.to_categorical(y_val)

    # Data augmentation
    augmentation = get_data_augmentation()

    # Callbacks: functions called during training

    # Early stopping: stop training if validation loss doesn't improve
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,           # Stop after 5 epochs without improvement
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint: save best model
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Reduce learning rate if training plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Train model
    history = model.fit(
        augmentation.flow(X_train, y_train_encoded, batch_size=batch_size),
        validation_data=(X_val, y_val_encoded),
        epochs=epochs,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    print("\n✅ Training completed!")

    return history


# ============================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set

    Args:
        model: trained keras model
        X_test: test images
        y_test: test labels

    Returns:
        test_loss, test_accuracy
    """

    print("\n" + "="*60)
    print("📊 EVALUATING ON TEST SET")
    print("="*60)

    y_test_encoded = keras.utils.to_categorical(y_test)

    test_loss, test_accuracy = model.evaluate(
        X_test, y_test_encoded, verbose=0)

    print(f"\n✅ Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

    return test_loss, test_accuracy


# ============================================================
# STEP 8: VISUALIZATION
# ============================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss over epochs
    """

    print("\n📈 Plotting training history...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], 'o-',
                 label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 's-',
                 label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Plot Loss
    axes[1].plot(history.history['loss'], 'o-',
                 label='Training Loss', color='red', linewidth=2)
    axes[1].plot(history.history['val_loss'], 's-',
                 label='Validation Loss', color='orange', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=100, bbox_inches='tight')
    print(f"✅ Saved plot to {HISTORY_PLOT_PATH}")
    plt.close(fig)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Main training pipeline
    """

    print("\n")
    print("█" * 60)
    print("█  PCB DEFECT DETECTION - MODEL TRAINING PIPELINE        █")
    print("█" * 60)

    parser = argparse.ArgumentParser(
        description="Train PCB defect classifier on uploaded dataset"
    )
    parser.add_argument(
        "--dataset-path",
        default=DATASET_PATH,
        help="Path to dataset root folder"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
        help="Image size for resizing"
    )
    args = parser.parse_args()

    # Step 1: Load dataset
    print("\n\n📂 STEP 1: LOADING DATASET")
    loader = DataLoader(args.dataset_path, args.img_size)
    X, y = loader.load_images_from_folder()

    # Step 2: Split dataset
    print("\n\n📂 STEP 2: SPLITTING DATASET")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)

    # Step 3: Build model
    print("\n\n🏗️  STEP 3: BUILDING MODEL")
    num_classes = len(loader.class_names)
    model = build_cnn_model(num_classes, input_shape=(
        args.img_size, args.img_size, 3))

    # Persist learned class order for inference scripts
    save_class_names(loader.class_names)
    save_model_metadata(args.img_size)

    # Step 4: Compile model
    print("\n\n⚙️  STEP 4: COMPILING MODEL")
    compile_model(model)

    # Step 5: Train model
    print("\n\n🚀 STEP 5: TRAINING MODEL")
    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=args.epochs, batch_size=args.batch_size)

    # Step 6: Evaluate on test set
    print("\n\n📊 STEP 6: EVALUATING ON TEST SET")
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)

    # Step 7: Plot training history
    print("\n\n📈 STEP 7: PLOTTING TRAINING HISTORY")
    plot_training_history(history)

    # Step 8: Summary
    print("\n\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"\n📊 Training Summary:")
    print(f"   Total Classes: {num_classes}")
    print(f"   Classes: {', '.join(loader.class_names)}")
    print(f"   Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Model Saved: {MODEL_SAVE_PATH}")
    print(f"   Plots Saved: {HISTORY_PLOT_PATH}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
