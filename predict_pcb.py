"""
PCB DEFECT DETECTION - PREDICTION SCRIPT
=========================================
This script loads the trained model and makes predictions
on new PCB images from the test dataset.

Usage:
    python predict_pcb.py --image path/to/pcb_image.jpg
    python predict_pcb.py --folder dataset/test/

Author: AI-ML Project
Date: 2024
"""

import os
import json
# pyright: reportMissingModuleSource=false
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import argparse
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/pcb_defect_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
MODEL_META_PATH = "models/model_meta.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_CSV = "output/predictions.csv"
DEFAULT_DATASET_FOLDER = "PCB_DATASET/images"

DEFAULT_DEFECT_CLASSES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper'
]

# Color codes for display
COLOR_MAP = {
    'missing_hole': 'red',
    'mouse_bite': 'orange',
    'open_circuit': 'yellow',
    'short': 'purple',
    'spur': 'blue',
    'spurious_copper': 'green'
}


# ============================================================
# PREDICTION CLASS
# ============================================================

class PCBDefectPredictor:
    """
    Make predictions on PCB images using trained model
    """

    def __init__(self, model_path=MODEL_PATH):
        """
        Load trained model

        Args:
            model_path: path to saved model .h5 file
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"🤖 Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully\n")

        self.img_size = self._load_img_size()
        self.defect_classes = self._load_class_names()
        self.predictions_list = []

    def _load_img_size(self):
        if os.path.exists(MODEL_META_PATH):
            try:
                with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                img_size = int(metadata.get("img_size", IMG_SIZE))
                print(f"✅ Loaded image size {img_size} from {MODEL_META_PATH}")
                return img_size
            except Exception as e:
                print(f"⚠️  Could not read {MODEL_META_PATH}: {e}")
        return IMG_SIZE

    def _load_class_names(self):
        if os.path.exists(CLASS_NAMES_PATH):
            try:
                with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                    class_names = json.load(f)
                if isinstance(class_names, list) and class_names:
                    print(f"✅ Loaded class names from {CLASS_NAMES_PATH}")
                    return class_names
            except Exception as e:
                print(f"⚠️  Could not read {CLASS_NAMES_PATH}: {e}")

        print("⚠️  Using default class names (class_names.json not found)")
        return DEFAULT_DEFECT_CLASSES

    def preprocess_image(self, image_path):
        """
        Load and preprocess image for prediction

        Args:
            image_path: path to PCB image

        Returns:
            preprocessed image or None if failed
        """

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return None

        # Resize to model input size
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert BGR to RGB (OpenCV uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values [0, 1]
        image = image.astype('float32') / 255.0

        return image

    def predict_single_image(self, image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
        """
        Predict defect on single PCB image

        Args:
            image_path: path to image
            confidence_threshold: minimum confidence threshold

        Returns:
            tuple: (result_dict_or_None, processed_image_or_None)
        """

        # Preprocess image
        image = self.preprocess_image(image_path)

        if image is None:
            return None, None

        # Prepare batch (add batch dimension)
        image_batch = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = self.model.predict(image_batch, verbose=0)

        # Get probabilities for all classes
        probabilities = predictions[0]

        # Get predicted class (highest probability)
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.defect_classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        # Create result dictionary
        result = {
            'image_name': os.path.basename(image_path),
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'confidence_value': confidence,
            'img_size': self.img_size
        }

        # Add probabilities for all classes
        for i, class_name in enumerate(self.defect_classes):
            result[f'prob_{class_name}'] = f"{probabilities[i]*100:.2f}%"

        return result, image  # Return image for visualization

    def predict_batch_from_folder(self, folder_path):
        """
        Predict on all images in a folder

        Args:
            folder_path: path to folder containing images

        Returns:
            list of prediction results
        """

        # Get all image files (recursive)
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return []

        print(f"\n📸 Found {len(image_files)} images to predict")
        print("="*70)

        predictions = []
        for i, image_path in enumerate(image_files, 1):
            result, _ = self.predict_single_image(image_path)

            if result is not None:
                predictions.append(result)

                # Display result
                print(f"\n{i}. {os.path.basename(image_path)}")
                print(f"   ➜ Prediction: {result['predicted_class']}")
                print(f"   ➜ Confidence: {result['confidence']}")
                print(f"   ➜ All probabilities:")
                for class_name in self.defect_classes:
                    prob = result[f'prob_{class_name}']
                    print(f"      • {class_name}: {prob}")

        print("\n" + "="*70)
        return predictions

    def save_predictions_csv(self, predictions, output_file=OUTPUT_CSV):
        """
        Save predictions to CSV file

        Args:
            predictions: list of prediction dictionaries
            output_file: output CSV file path
        """

        if not predictions:
            print("⚠️  No predictions to save")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(predictions)

        # Select relevant columns
        columns = ['image_name', 'predicted_class', 'confidence'] + \
            [f'prob_{c}' for c in self.defect_classes]

        df_output = df[columns]

        # Save to CSV
        df_output.to_csv(output_file, index=False)

        print(f"✅ Predictions saved to {output_file}")


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def visualize_single_prediction(image_path, result, defect_classes):
    """
    Display prediction on image with visualization

    Args:
        image_path: path to image
        result: prediction result dict
        model: model for visualization
    """

    # Preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image for visualization: {image_path}")
        return

    display_size = int(result.get('img_size', IMG_SIZE))
    image = cv2.resize(image, (display_size, display_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Original image with prediction text
    axes[0].imshow(image)
    axes[0].set_title(
        f"PCB Image: {result['image_name']}", fontsize=12, fontweight='bold')

    # Add text with results
    textstr = f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[0].text(0.5, 0.05, textstr, transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='center', bbox=props)
    axes[0].axis('off')

    # Right: Bar chart of all class probabilities
    probabilities = []
    for class_name in defect_classes:
        prob_str = result[f'prob_{class_name}'].replace('%', '')
        probabilities.append(float(prob_str))

    colors = [COLOR_MAP.get(c, 'gray') for c in defect_classes]
    bars = axes[1].barh(defect_classes, probabilities, color=colors, alpha=0.7)
    axes[1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1].set_title('Class Probabilities', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 100])

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        axes[1].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_batch_predictions(predictions, images_folder=None, num_display=9):
    """
    Display multiple predictions in a grid

    Args:
        predictions: list of prediction results
        images_folder: folder path for images
        num_display: number of images to display
    """

    num_display = min(num_display, len(predictions))

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_display)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_display):
        result = predictions[i]
        image_path = result['image_path']
        display_size = int(result.get('img_size', IMG_SIZE))

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            axes[i].set_title("Image load failed", color='red',
                              fontweight='bold', fontsize=10)
            axes[i].axis('off')
            continue

        image = cv2.resize(image, (display_size, display_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image
        axes[i].imshow(image)

        # Color based on prediction confidence
        if float(result['confidence_value']) > 0.8:
            color = 'green'
        elif float(result['confidence_value']) > 0.6:
            color = 'orange'
        else:
            color = 'red'

        # Title with prediction
        title = f"{result['predicted_class']}\n{result['confidence']}"
        axes[i].set_title(title, color=color, fontweight='bold', fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_display, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """
    Main prediction pipeline with CLI
    """

    print("\n")
    print("█" * 70)
    print("█  PCB DEFECT DETECTION - PREDICTION & CLASSIFICATION PIPELINE      █")
    print("█" * 70)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict PCB defects using trained CNN model'
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to single PCB image for prediction'
    )

    parser.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing PCB images'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Display visualizations'
    )

    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save predictions to CSV'
    )

    args = parser.parse_args()

    # Create predictor
    print("\n🤖 INITIALIZING PREDICTOR")
    print("="*70)
    predictor = PCBDefectPredictor(MODEL_PATH)

    # Single image prediction
    if args.image:
        print(f"\n📸 PREDICTING SINGLE IMAGE")
        print("="*70)
        print(f"Image: {args.image}\n")

        result, image = predictor.predict_single_image(args.image)

        if result:
            print(f"✅ Prediction: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"\n   All probabilities:")
            for class_name in predictor.defect_classes:
                prob = result[f'prob_{class_name}']
                print(f"   • {class_name}: {prob}")

            if args.visualize:
                visualize_single_prediction(
                    args.image, result, predictor.defect_classes)

    # Batch prediction from folder
    elif args.folder:
        print(f"\n📂 BATCH PREDICTION FROM FOLDER")
        print("="*70)
        print(f"Folder: {args.folder}\n")

        predictions = predictor.predict_batch_from_folder(args.folder)

        if predictions and args.save_csv:
            predictor.save_predictions_csv(predictions)

        if predictions and args.visualize:
            visualize_batch_predictions(predictions)

    # Default: run on attached dataset images folder
    else:
        print(f"\n📂 DEFAULT: TESTING ON ATTACHED DATASET")
        print("="*70)

        test_folder = DEFAULT_DATASET_FOLDER
        if os.path.exists(test_folder):
            predictions = predictor.predict_batch_from_folder(test_folder)

            # Always save CSV for default case
            predictor.save_predictions_csv(predictions)
        else:
            print(f"\n⚠️  Test folder not found: {test_folder}")
            print("   Usage examples:")
            print("   python predict_pcb.py --image path/to/image.jpg")
            print("   python predict_pcb.py --folder path/to/dataset/")
            print(
                "   python predict_pcb.py --folder PCB_DATASET/images --save-csv --visualize")


if __name__ == "__main__":
    main()
