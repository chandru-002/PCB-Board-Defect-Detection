"""
PCB DEFECT DETECTION - AUTOMATED SETUP SCRIPT
==============================================
One-click setup for the entire project

This script will:
1. ✅ Install Python dependencies
2. ✅ Download dataset from Kaggle
3. ✅ Organize dataset structure
4. ✅ Verify everything is ready
5. ✅ Start training

Usage:
    python setup.py

Author: AI-ML Project
Date: 2024
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_header(title):
    """Print formatted header"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║ " + title.center(66) + " ║")
    print("╚" + "═"*68 + "╝")


def print_step(step_num, title):
    """Print step header"""
    print(f"\n\n{'▶'*2} STEP {step_num}: {title}")
    print("═" * 70)


def run_command(command, description=""):
    """
    Run a shell command

    Args:
        command: command to run
        description: description of what command does

    Returns:
        True if successful, False otherwise
    """
    try:
        if description:
            print(f"\n📋 {description}")

        result = subprocess.run(command, shell=True,
                                capture_output=True, text=True)

        if result.returncode == 0:
            if description:
                print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info

    print(
        f"\n   Python Version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False

    print("✅ Python version compatible")
    return True


def install_dependencies():
    """Install Python packages from requirements.txt"""
    print("\n   Installing packages: tensorflow, opencv, numpy, pandas...")

    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False

    if run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("✅ All dependencies installed")
        return True
    else:
        print("⚠️  Some packages may not have installed")
        print("   Try: pip install -r requirements.txt")
        return False


def check_kaggle_setup():
    """Check if Kaggle API is set up"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_api_token = os.environ.get("KAGGLE_API_TOKEN")

    if kaggle_json.exists():
        print("\n✅ Kaggle API credentials found")
        return True
    elif kaggle_api_token:
        print("\n✅ Kaggle API token found in environment")
        return True
    else:
        print("\n⚠️  Kaggle API credentials NOT found")
        print("\n   To set up:")
        print("   1. Go to: https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Move kaggle.json to: ~/.kaggle/")
        print("   4. Or set env var: KAGGLE_API_TOKEN")
        print("\n   After setup, run this script again")
        return False


def download_dataset():
    """Download dataset from Kaggle"""
    if not os.path.exists("dataset"):
        print("\n📥 Downloading PCB defects dataset from Kaggle...")
        print("   Dataset: akhatova/pcb-defects")

        if run_command(
            f"{sys.executable} download_dataset.py",
            "Downloading and organizing dataset"
        ):
            print("✅ Dataset downloaded successfully")
            return True
        else:
            print("❌ Failed to download dataset")
            return False
    else:
        print("\n✅ Dataset folder already exists")
        return True


def verify_dataset():
    """Verify dataset structure"""
    print("\n   Checking dataset structure...")

    required_classes = [
        'missing_hole',
        'mouse_bite',
        'open_circuit',
        'short',
        'spur',
        'spurious_copper'
    ]

    all_found = True
    total_images = 0

    for class_name in required_classes:
        class_path = os.path.join("dataset", class_name)

        if os.path.exists(class_path):
            image_count = len([f for f in os.listdir(class_path)
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   ✅ {class_name}: {image_count} images")
            total_images += image_count
        else:
            print(f"   ❌ {class_name}: NOT FOUND")
            all_found = False

    if all_found and total_images > 0:
        print(f"\n✅ Dataset verified ({total_images} total images)")
        return True
    else:
        print("\n❌ Dataset verification failed")
        return False


def verify_installation():
    """Verify all packages are installed"""
    print("\n   Checking Python packages...")

    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }

    all_installed = True

    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - NOT INSTALLED")
            all_installed = False

    if all_installed:
        print("\n✅ All packages installed")
        return True
    else:
        print("\n❌ Some packages missing")
        return False


def show_next_steps():
    """Show next steps for user"""
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)

    print("\n📚 NEXT STEPS:\n")

    print("1️⃣  Explore your dataset (optional):")
    print("    python explore_dataset.py\n")

    print("2️⃣  Train the model:")
    print("    python train_model.py\n")

    print("3️⃣  Make predictions:")
    print("    python predict_pcb.py\n")

    print("📊 Results will be saved to:")
    print("   • models/pcb_defect_model.h5")
    print("   • output/training_history.png")
    print("   • output/predictions.csv\n")


def main():
    """Main setup pipeline"""

    print_header("🚀 PCB DEFECT DETECTION - AUTOMATED SETUP")

    # Step 1: Check Python
    print_step(1, "Checking Python Installation")
    if not check_python_version():
        print("\n❌ Setup failed")
        sys.exit(1)

    # Step 2: Install dependencies
    print_step(2, "Installing Python Packages")
    if not install_dependencies():
        print("\n⚠️  Continue? (y/n): ", end="")
        if input().lower() != 'y':
            sys.exit(1)

    # Step 3: Check Kaggle credentials
    print_step(3, "Checking Kaggle API Credentials")
    if not check_kaggle_setup():
        print("\n⏭️  Skipping dataset download")
        print("ℹ️  You can download manually and place in 'dataset/' folder")
    else:
        # Step 4: Download dataset
        print_step(4, "Downloading Dataset from Kaggle")
        if not download_dataset():
            print("\n⚠️  Continue without dataset? (y/n): ", end="")
            if input().lower() != 'y':
                sys.exit(1)

    # Step 5: Verify setup
    print_step(5, "Verifying Installation")

    if not verify_installation():
        print("\n❌ Installation verification failed")
        sys.exit(1)

    if os.path.exists("dataset"):
        if not verify_dataset():
            print("\n⚠️  Dataset verification failed")
            print("   Make sure dataset is in correct location")
    else:
        print("\n⚠️  Dataset folder not found")
        print("   Download from Kaggle or run download_dataset.py")

    # Step 6: Show next steps
    print_step(6, "Setup Complete - Next Steps")
    show_next_steps()

    print("="*70)
    print("✅ Setup successful! Ready to train your model!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
