"""
PCB DEFECT DETECTION - DATASET EXPLORER
========================================
Explore and analyze your dataset before training.

Usage:
    python explore_dataset.py

Author: AI-ML Project
Date: 2024
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
DATASET_PATH = "dataset"
SAMPLE_IMAGES_PER_CLASS = 3
IMG_SIZE = 224


class DatasetExplorer:
    """
    Explore PCB defect dataset
    """

    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.class_data = {}

    def analyze_dataset(self):
        """
        Analyze dataset structure and statistics
        """

        print("\n" + "="*70)
        print("📊 PCB DEFECT DATASET ANALYSIS")
        print("="*70)

        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset folder not found: {self.dataset_path}")
            print("   Please download dataset from Kaggle first!")
            return None

        # Get all class folders
        class_folders = [f for f in os.listdir(self.dataset_path)
                         if os.path.isdir(os.path.join(self.dataset_path, f))]

        class_folders.sort()

        print(f"\n✅ Found {len(class_folders)} defect classes:\n")

        total_images = 0
        size_info = "Unknown"

        # Analyze each class
        for idx, class_name in enumerate(class_folders, 1):
            class_path = os.path.join(self.dataset_path, class_name)

            # Get all image files
            image_files = [f for f in os.listdir(class_path)
                           if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]

            image_count = len(image_files)
            total_images += image_count

            # Load one image to get dimensions
            if image_files:
                sample_path = os.path.join(class_path, image_files[0])
                sample_image = cv2.imread(sample_path)

                if sample_image is not None:
                    height, width, channels = sample_image.shape
                    size_info = f"{width}×{height}×{channels}"
                else:
                    size_info = "Unknown"
            else:
                size_info = "No images"

            # Store data
            self.class_data[class_name] = {
                'count': image_count,
                'path': class_path,
                'files': image_files,
                'size': size_info
            }

            # Print class info
            percentage = (image_count / (total_images + 0.1)) * \
                100 if total_images > 0 else 0
            bar = "█" * (image_count // 5) + "░" * (40 - (image_count // 5))

            print(f"   {idx}. {class_name:20s} │ {bar} │ {image_count:4d} images")

        print(f"\n{'─'*70}")
        print(f"   Total: {total_images} images")
        print(f"   Classes: {len(class_folders)}")
        print(f"   Image Size: {size_info}")
        print(f"{'─'*70}")

        # Check balance
        print(f"\n📈 Dataset Balance:")
        if total_images > 0:
            for class_name in sorted(self.class_data.keys()):
                count = self.class_data[class_name]['count']
                percentage = (count / total_images) * 100
                status = "✅" if 10 < percentage < 20 else "⚠️ "
                print(
                    f"   {status} {class_name:20s}: {count:4d} ({percentage:5.1f}%)")

        return total_images > 0

    def visualize_samples(self):
        """
        Display sample images from each class
        """

        if not self.class_data:
            print("❌ No dataset found. Run analyze_dataset() first.")
            return

        num_classes = len(self.class_data)

        print(f"\n\n🖼️  DISPLAYING SAMPLE IMAGES")
        print("="*70)
        print(f"Showing {SAMPLE_IMAGES_PER_CLASS} sample images per class\n")

        fig, axes = plt.subplots(num_classes, SAMPLE_IMAGES_PER_CLASS,
                                 figsize=(12, 2*num_classes))

        if num_classes == 1:
            axes = axes.reshape(1, -1)

        for row, (class_name, class_info) in enumerate(sorted(self.class_data.items())):
            print(f"   {row+1}. {class_name}:")

            image_files = class_info['files']

            for col in range(SAMPLE_IMAGES_PER_CLASS):
                if col < len(image_files):
                    image_path = os.path.join(
                        class_info['path'], image_files[col])
                    image = cv2.imread(image_path)
                    if image is None:
                        axes[row, col].set_title("Load failed", fontsize=8)
                        axes[row, col].axis('off')
                        print(f"      ⚠️  Could not read: {image_files[col]}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    axes[row, col].imshow(image)
                    axes[row, col].set_title(
                        f"{image_files[col][:15]}...", fontsize=8)
                    axes[row, col].axis('off')

                    print(f"      • {image_files[col]}")
                else:
                    axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig("output/dataset_samples.png", dpi=100, bbox_inches='tight')
        print(f"\n✅ Saved sample images to: output/dataset_samples.png")
        plt.show()

    def check_image_quality(self):
        """
        Check for corrupted or problematic images
        """

        print(f"\n\n✅ CHECKING IMAGE QUALITY")
        print("="*70)

        corrupted_files = []
        total_checked = 0

        for class_name, class_info in self.class_data.items():
            print(f"\n   Checking {class_name}...")

            for image_file in class_info['files']:
                image_path = os.path.join(class_info['path'], image_file)
                total_checked += 1

                try:
                    image = cv2.imread(image_path)

                    if image is None:
                        corrupted_files.append(
                            (class_name, image_file, "Cannot read file"))
                        print(f"      ❌ {image_file}: Cannot read")
                    elif image.size == 0:
                        corrupted_files.append(
                            (class_name, image_file, "Empty image"))
                        print(f"      ❌ {image_file}: Empty image")
                    elif len(image.shape) != 3:
                        corrupted_files.append(
                            (class_name, image_file, "Invalid dimensions"))
                        print(f"      ❌ {image_file}: Invalid dimensions")

                except Exception as e:
                    corrupted_files.append((class_name, image_file, str(e)))
                    print(f"      ❌ {image_file}: Error - {str(e)}")

        print(f"\n{'─'*70}")
        print(f"   Total files checked: {total_checked}")
        print(f"   Corrupted files: {len(corrupted_files)}")

        if corrupted_files:
            print(f"\n⚠️  CORRUPTED FILES FOUND:")
            for class_name, filename, reason in corrupted_files:
                print(f"   • {class_name}/{filename} - {reason}")
        else:
            print(f"\n✅ All images are valid!")

        return len(corrupted_files) == 0

    def get_statistics(self):
        """
        Get dataset statistics
        """

        print(f"\n\n📊 DATASET STATISTICS")
        print("="*70)

        total_images = sum(info['count'] for info in self.class_data.values())
        avg_images = total_images / \
            len(self.class_data) if self.class_data else 0
        min_images = min(
            info['count'] for info in self.class_data.values()) if self.class_data else 0
        max_images = max(
            info['count'] for info in self.class_data.values()) if self.class_data else 0

        print(f"\n   Total Images: {total_images}")
        print(f"   Total Classes: {len(self.class_data)}")
        print(f"   Average per class: {avg_images:.0f}")
        print(f"   Min images: {min_images}")
        print(f"   Max images: {max_images}")
        print(
            f"   Imbalance ratio: {max_images/min_images:.2f}x" if min_images > 0 else "")

        return {
            'total_images': total_images,
            'num_classes': len(self.class_data),
            'avg_per_class': avg_images,
            'min_images': min_images,
            'max_images': max_images
        }

    def generate_report(self):
        """
        Generate complete dataset report
        """

        print("\n\n" + "="*70)
        print("📋 DATASET EXPLORER REPORT")
        print("="*70)

        # Analyze
        self.analyze_dataset()

        # Check quality
        self.check_image_quality()

        # Statistics
        stats = self.get_statistics()

        # Visualize
        try:
            self.visualize_samples()
        except Exception as e:
            print(f"⚠️  Could not visualize samples: {str(e)}")

        # Recommendations
        print(f"\n\n💡 RECOMMENDATIONS FOR TRAINING")
        print("="*70)

        if stats['total_images'] < 300:
            print(f"⚠️  Dataset is small ({stats['total_images']} images)")
            print(f"    Recommendation: Collect more images or use data augmentation")
            print(f"    Target: 400-500 images")
        elif stats['total_images'] < 500:
            print(f"⚠️  Dataset is moderate ({stats['total_images']} images)")
            print(f"    Recommendation: Good for training, adds more for better accuracy")
        else:
            print(f"✅ Dataset is large ({stats['total_images']} images)")
            print(f"    Recommendation: Excellent for training deep learning models")

        imbalance = stats['max_images'] / \
            stats['min_images'] if stats['min_images'] > 0 else 1

        if imbalance > 2:
            print(f"\n⚠️  Dataset is imbalanced (ratio: {imbalance:.2f}x)")
            print(
                f"    Recommendation: Use class weights or collect more for minority classes")
        else:
            print(f"\n✅ Dataset is well-balanced (ratio: {imbalance:.2f}x)")

        print(f"\n{'─'*70}")
        print("✅ REPORT COMPLETE!")
        print("="*70)


def main():
    """
    Main explorer pipeline
    """

    print("\n")
    print("╔═" + "="*68 + "═╗")
    print("║  " + " "*66 + "  ║")
    print("║  " + "🔍 PCB DEFECT DETECTION - DATASET EXPLORER TOOL 🔍".center(66) + "  ║")
    print("║  " + " "*66 + "  ║")
    print("╚═" + "="*68 + "═╝")

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Create explorer
    explorer = DatasetExplorer(DATASET_PATH)

    # Generate report
    explorer.generate_report()

    print("\n\n🎓 Next Steps:")
    print("   1. Review the dataset report above")
    print("   2. Check output/dataset_samples.png for sample images")
    print("   3. If everything looks good, run: python train_model.py")
    print("\n")


if __name__ == "__main__":
    main()
