"""
PCB DEFECT DETECTION - KAGGLE DATASET DOWNLOADER
=================================================
Automatically download PCB defects dataset from Kaggle using kagglehub

Dataset: akhatova/pcb-defects
Source: https://www.kaggle.com/datasets/akhatova/pcb-defects

Installation:
    pip install kagglehub

Authentication:
    1. Go to Kaggle.com → Settings → API
    2. Click "Create New API Token"
    3. File will be saved as ~/.kaggle/kaggle.json
    4. Run this script to download

Usage:
    python download_dataset.py

Author: AI-ML Project
Date: 2024
"""

import os
import sys
import shutil
from pathlib import Path


def install_kagglehub():
    """
    Install kagglehub if not already installed
    """
    try:
        import kagglehub
        print("✅ kagglehub already installed")
        return True
    except ImportError:
        print("📦 Installing kagglehub...")
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "kagglehub"])
            print("✅ kagglehub installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install kagglehub: {e}")
            return False


def check_kaggle_credentials():
    """
    Check if Kaggle API credentials exist
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_api_token = os.environ.get("KAGGLE_API_TOKEN")

    if kaggle_json.exists():
        print("✅ Kaggle credentials found in ~/.kaggle/kaggle.json")
        return True

    if kaggle_api_token:
        print("✅ Kaggle credentials found in KAGGLE_API_TOKEN")
        return True

    if not kaggle_json.exists():
        print("\n" + "="*70)
        print("❌ KAGGLE CREDENTIALS NOT FOUND")
        print("="*70)
        print("\nTo download from Kaggle, you need to authenticate:")
        print("\n1. Go to: https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json to your Downloads folder")
        print("5. Move it to: ~/.kaggle/kaggle.json")
        print("   (usually: C:\\Users\\<username>\\.kaggle\\kaggle.json on Windows)")
        print("6. Or set an environment variable:")
        print("   PowerShell: $env:KAGGLE_API_TOKEN='your_token_here'")
        print("7. Run this script again")
        print("="*70)
        return False

    return True


def download_dataset():
    """
    Download PCB defects dataset from Kaggle using kagglehub
    """
    import kagglehub

    print("\n" + "="*70)
    print("📥 DOWNLOADING PCB DEFECTS DATASET FROM KAGGLE")
    print("="*70)

    try:
        # Dataset path on Kaggle
        dataset_path = "akhatova/pcb-defects"

        print(f"\n🔗 Dataset: {dataset_path}")
        print("⏳ Downloading... (this may take a few minutes)")

        # Download dataset
        path = kagglehub.dataset_download(dataset_path)

        print(f"\n✅ Dataset downloaded to: {path}")

        return path

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return None


def organize_dataset(download_path, target_path="dataset"):
    """
    Organize downloaded dataset into proper folder structure

    Expected structure:
    dataset/
    ├── missing_hole/
    ├── mouse_bite/
    ├── open_circuit/
    ├── short/
    ├── spur/
    └── spurious_copper/
    """

    print("\n" + "="*70)
    print("📂 ORGANIZING DATASET")
    print("="*70)

    os.makedirs(target_path, exist_ok=True)

    try:
        # Find image files in download path
        defect_classes = {}

        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    # Try to determine class from folder name
                    folder_name = os.path.basename(root).lower()

                    # Map folder names to classes
                    if 'missing' in folder_name or 'hole' in folder_name:
                        class_name = 'missing_hole'
                    elif 'mouse' in folder_name or 'bite' in folder_name:
                        class_name = 'mouse_bite'
                    elif 'open' in folder_name:
                        class_name = 'open_circuit'
                    elif 'short' in folder_name:
                        class_name = 'short'
                    elif 'spur' in folder_name:
                        class_name = 'spur'
                    elif 'spurious' in folder_name or 'copper' in folder_name:
                        class_name = 'spurious_copper'
                    else:
                        class_name = folder_name

                    if class_name not in defect_classes:
                        defect_classes[class_name] = []

                    defect_classes[class_name].append(os.path.join(root, file))

        # Create class folders and copy images
        total_images = 0

        for class_name, image_files in defect_classes.items():
            class_path = os.path.join(target_path, class_name)
            os.makedirs(class_path, exist_ok=True)

            for image_file in image_files:
                try:
                    dest_file = os.path.join(
                        class_path, os.path.basename(image_file))
                    shutil.copy2(image_file, dest_file)
                    total_images += 1
                except Exception as e:
                    print(f"⚠️  Failed to copy {image_file}: {e}")

        print(f"\n✅ Dataset organized successfully!")
        print(f"\n📊 Dataset Structure:")
        for class_name in sorted(defect_classes.keys()):
            count = len([f for f in os.listdir(os.path.join(target_path, class_name))
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   • {class_name}: {count} images")

        print(f"\n   Total images: {total_images}")

        return total_images > 0

    except Exception as e:
        print(f"\n❌ Error organizing dataset: {e}")
        return False


def main():
    """
    Main dataset download pipeline
    """

    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║  🔽 PCB DEFECT DETECTION - KAGGLE DATASET DOWNLOADER 🔽       ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # Step 1: Install kagglehub
    print("\n\n📦 STEP 1: CHECKING KAGGLEHUB")
    print("="*70)
    if not install_kagglehub():
        print("\n❌ Failed to install kagglehub")
        sys.exit(1)

    # Step 2: Check credentials
    print("\n\n🔑 STEP 2: CHECKING KAGGLE CREDENTIALS")
    print("="*70)
    if not check_kaggle_credentials():
        print("\n📝 Please set up Kaggle API credentials first")
        sys.exit(1)

    # Step 3: Download dataset
    print("\n\n📥 STEP 3: DOWNLOADING DATASET")
    print("="*70)
    download_path = download_dataset()

    if not download_path:
        print("\n❌ Failed to download dataset")
        sys.exit(1)

    # Step 4: Organize dataset
    print("\n\n📂 STEP 4: ORGANIZING DATASET")
    print("="*70)
    if organize_dataset(download_path):
        print("\n" + "="*70)
        print("✅ DATASET DOWNLOAD & ORGANIZATION COMPLETE!")
        print("="*70)
        print("\n✅ Ready to train! Run:")
        print("   python train_model.py")
        print("\n")
    else:
        print("\n❌ Failed to organize dataset")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
