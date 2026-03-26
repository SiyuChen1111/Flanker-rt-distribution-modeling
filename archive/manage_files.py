#!/usr/bin/env python3
"""
File Management Script for VAM Study Project
Organizes generated files into appropriate directories
"""

import os
import shutil
from pathlib import Path

# Define directories
BASE_DIR = Path("/Users/siyu/Documents/GitHub/VAM-studying")
ARCHIVE_DIR = BASE_DIR / "archive"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Create directories
for dir_path in [ARCHIVE_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR]:
    dir_path.mkdir(exist_ok=True)
    print(f"✓ Created directory: {dir_path}")

# Core files to keep in root
CORE_FILES = [
    "logs.md",
    "siyu_study.md",
    "vgg_drift_rate_complete.ipynb",
    "vgg_drift_rate_fixed.ipynb",
    "README.md",
]

# Notebooks to move
NOTEBOOKS = [
    "flanker_vgg_classification.ipynb",
    "flanker_vgg_classification_corrected.ipynb",
]

# Scripts to archive
SCRIPTS_TO_ARCHIVE = [
    "download_vgg16.py",
    "vgg16_download_alternatives.py",
    "test_vgg16_loading.py",
    "test_vgg_dimensions.py",
    "fix_vgg_dimensions.py",
    "fix_notebook_complete.py",
    "add_visualization.py",
    "fix_visualization.py",
    "update_with_real_birds.py",
    "add_shape_display.py",
    "manage_files.py",
]

print("\n" + "="*80)
print("File Management Summary")
print("="*80)

# Move notebooks
print("\n📓 Moving Notebooks:")
for file in NOTEBOOKS:
    src = BASE_DIR / file
    dst = NOTEBOOKS_DIR / file
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  ✓ Moved {file} → notebooks/")

# Archive scripts
print("\n📦 Archiving Scripts:")
for file in SCRIPTS_TO_ARCHIVE:
    src = BASE_DIR / file
    dst = ARCHIVE_DIR / file
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  ✓ Moved {file} → archive/")

# Check core files
print("\n📋 Core Files (kept in root):")
for file in CORE_FILES:
    if (BASE_DIR / file).exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ⚠ {file} (not found)")

print("\n" + "="*80)
print("File organization complete!")
print("="*80)
print("\nDirectory Structure:")
print("  /VAM-studying/")
print("    ├── logs.md                    # Study log")
print("    ├── siyu_study.md              # Learning notes")
print("    ├── vgg_drift_rate_complete.ipynb  # Main notebook")
print("    ├── vgg_drift_rate_fixed.ipynb     # Fixed version")
print("    ├── notebooks/                 # Additional notebooks")
print("    ├── archive/                   # Helper scripts")
print("    └── vam/                       # Original VAM code")
