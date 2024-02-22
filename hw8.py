import os
import shutil
import numpy as np

# Define the path to the full dataset and the prototype dataset
full_dataset_path = 'path/to/full_dataset'
prototype_dataset_path = 'path/to/prototype_dataset'

# Define the split ratios
train_ratio = 0.7
validate_ratio = 0.15
test_ratio = 0.15

# Make sure the prototype dataset path exists
if not os.path.exists(prototype_dataset_path):
    os.makedirs(prototype_dataset_path)

# Create train, validate, and test directories
for split in ['train', 'validate', 'test']:
    split_path = os.path.join(prototype_dataset_path, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

# Distribute images into the prototype dataset
for category in os.listdir(full_dataset_path):
    # Create category subdirectories in train, validate, test directories
    for split in ['train', 'validate', 'test']:
        category_path = os.path.join(prototype_dataset_path, split, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # List all images in the full dataset category directory
    category_full_path = os.path.join(full_dataset_path, category)
    images = os.listdir(category_full_path)
    
    # Shuffle images for random distribution
    np.random.shuffle(images)

    # Calculate the number of images for each set
    train_count = int(train_ratio * len(images))
    validate_count = int(validate_ratio * len(images))
    test_count = len(images) - train_count - validate_count

    # Move images to the respective directories
    for i, image in enumerate(images):
        source_path = os.path.join(category_full_path, image)
        
        if i < train_count:
            target_path = os.path.join(prototype_dataset_path, 'train', category, image)
        elif i < train_count + validate_count:
            target_path = os.path.join(prototype_dataset_path, 'validate', category, image)
        else:
            target_path = os.path.join(prototype_dataset_path, 'test', category, image)
        
        shutil.copy(source_path, target_path)

