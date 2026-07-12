import os
import cv2

def load_data(data_path):
    """
    Loads images and their corresponding emotion labels from a given directory path.
    Assumes the directory contains subfolders named by category.
    """
    # Get the list of emotion folders and sort them to keep labels consistent
    folder_list = os.listdir(data_path)
    folder_list.sort()
    
    print(f"Loading data from {data_path}")
    print("Categories found:", folder_list)
    
    X = []
    y = []
    
    # Load the data into arrays
    for i, category in enumerate(folder_list):
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            continue
            
        files = os.listdir(category_path)
        for file in files:
            img_path = os.path.join(category_path, file)
            # Read image in grayscale mode (0)
            img = cv2.imread(img_path, 0)
            if img is not None:
                X.append(img)
                y.append(i)
                
    print(f"Total images loaded: {len(X)}")
    return X, y
