import pandas as pd
import math
import cv2
import os
import shutil

def get_coordinates_from_filename(filename):
    # Extract the coordinates part from filename
    # e.g., from "29-17-IIDC.svs_[8000,24000,4000,4000]_23.png"
    # get 8000,24000
    coords_part = filename.split('[')[1].split(']')[0]
    x, y = map(int, coords_part.split(',')[:2])
    return x, y

def point_belongs_to_image(point_x, point_y, image_x, image_y):
    # Swap x and y when checking containment
    return (image_y <= point_x < image_y + 4000 and 
            image_x <= point_y < image_x + 4000)

# Create output directory if it doesn't exist
output_dir = 'images_with_attention_points'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the points data
# 29-17-IIDC.xlsx
# D-001-18.csv
# S-18-4594.csv
# S-19-126H.csv

session_name = "29-17-IIDC"

if session_name == "29-17-IIDC":
    df = pd.read_excel('original_data/gaze_labelled/raw/' + session_name + '.xlsx', header=None)
else:
    df = pd.read_csv('original_data/gaze_labelled/raw/' + session_name + '.csv', header=None)

points = list(zip(df[0], df[1]))  # Convert to list of (x,y) tuples

# Process each image in the train directory
train_dir = 'original_data/gaze_labelled/train'
count_processed = 0
count_original = 0

for filename in os.listdir(train_dir):
    if filename.startswith(session_name) and filename.endswith('.png'):
        count_original += 1
        # Get image coordinates from filename
        image_x, image_y = get_coordinates_from_filename(filename)
        
        # Find points that belong to this image
        image_points = []
        for point_x, point_y in points:
            if point_belongs_to_image(point_x, point_y, image_x, image_y):
                # Convert to relative coordinates within the image - swap x and y here too
                relative_x = int(point_x - image_y)
                relative_y = int(point_y - image_x)
                image_points.append((relative_x, relative_y))
        
        if image_points:  # Only process images that have points
            count_processed += 1
            print(f"\nProcessing {filename}")
            
            # Read the image
            input_path = os.path.join(train_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = cv2.imread(input_path)
            
            # Draw points
            for point in image_points:
                cv2.circle(img, point, radius=10, color=(0, 255, 0), thickness=-1)  # Red filled circle
                cv2.circle(img, point, radius=10, color=(0, 0, 0), thickness=2)     # Black border
            
            # Save the image with points
            cv2.imwrite(output_path, img)
            print(f"Saved with {len(image_points)} points")
print(f"{count_processed}/{count_original} Images processed!")
print("\nProcessing complete!")
