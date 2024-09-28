import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import json

def binarize_image(image_path, output_path):
    print(f"Step 1: Binarizing the image {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply threshold to isolate the dark circles
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imwrite(output_path, binary)
    print(f"Binarized image saved to {output_path}")
    return binary

def label_components(binary_image, output_path):
    print("Step 2: Finding and labeling connected components...")
    
    # Ensure the image is binary (0 and 255)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the image if necessary (assuming black circles on white background)
    if np.mean(binary_image) > 127:
        binary_image = cv2.bitwise_not(binary_image)
    
    # Apply connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Filter components based on area to remove small noise
    min_area = 50  # Adjust this value based on the size of your smallest circle
    valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    
    num_components = len(valid_labels)
    print(f"Found {num_components} valid connected components")

    # Create a colored image to visualize components
    colored_labels = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    
    print("Adding labels to components...")
    for i, label in enumerate(valid_labels, start=1):
        # Generate a random color for this component
        color = np.random.randint(0, 255, 3).tolist()
        
        # Color the component
        colored_labels[labels == label] = color
        
        # Get the centroid
        centroid = centroids[label].astype(int)
        
        # Add label with adjusted font size
        cv2.putText(colored_labels, str(i), tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 0, 0), 9)
        cv2.putText(colored_labels, str(i), tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 255, 255), 6)

    cv2.imwrite(output_path, colored_labels)
    print(f"Labeled image saved to {output_path}")
    return labels, num_components, valid_labels, centroids, colored_labels

def fit_gmm_to_components(labeled_image, num_components, valid_labels, centroids, colored_labels, output_path):
    print("Step 3: Fitting GMM to each component...")
    gmm_params = {}
    
    # Create a copy of the colored labels image for visualization
    vis_image = colored_labels.copy()
    
    for i, label in enumerate(valid_labels, start=1):
        print(f"Fitting GMM for component {i}/{num_components}")
        y, x = np.where(labeled_image == label)
        points = np.column_stack((x, y))
        
        gmm = GaussianMixture(n_components=1, random_state=42)
        gmm.fit(points)
        
        # Store GMM parameters
        weights = gmm.weights_[0]
        mean = gmm.means_[0]
        cov = gmm.covariances_[0]
        
        gmm_params[f'component_{i}'] = {
            'weights': weights.tolist(),
            'means': mean.tolist(),
            'covariances': cov.tolist()
        }
        
        # Write component number on the image
        centroid = centroids[label].astype(int)
        cv2.putText(vis_image, f"{i}", tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 0, 0), 9)
        cv2.putText(vis_image, f"{i}", tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 255, 255), 6)
    
    # Write overall GMM parameters at the top left
    y_offset = 60
    for i, params in gmm_params.items():
        mean = params['means']
        cov = params['covariances']
        text = f"{i}: mean=({mean[0]:.1f}, {mean[1]:.1f}) cov=({cov[0][0]:.1f}, {cov[0][1]:.1f}, {cov[1][0]:.1f}, {cov[1][1]:.1f})"
        cv2.putText(vis_image, text, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 9)
        cv2.putText(vis_image, text, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6)
        y_offset += 100
    
    cv2.imwrite(output_path, vis_image)
    print(f"Image with GMM parameters saved to {output_path}")
    
    return gmm_params

def process_image(image_path, output_folder):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output folders
    binary_folder = os.path.join(output_folder, "binary_images")
    labeled_folder = os.path.join(output_folder, "labeled_images")
    gmm_folder = os.path.join(output_folder, "gmm_images")
    os.makedirs(binary_folder, exist_ok=True)
    os.makedirs(labeled_folder, exist_ok=True)
    os.makedirs(gmm_folder, exist_ok=True)
    
    # Define output paths
    binary_output = os.path.join(binary_folder, f"{image_name}_binary.png")
    labeled_output = os.path.join(labeled_folder, f"{image_name}_labeled.png")
    gmm_output = os.path.join(gmm_folder, f"{image_name}_gmm.png")

    # Step 1: Binarize image
    binary_img = binarize_image(image_path, binary_output)

    # Step 2: Label connected components
    labeled_img, num_components, valid_labels, centroids, colored_labels = label_components(binary_img, labeled_output)

    # Step 3: Fit GMM to each component and save parameters
    gmm_params = fit_gmm_to_components(labeled_img, num_components, valid_labels, centroids, colored_labels, gmm_output)

    print(f"Processing complete for {image_name}. Found {num_components} components.")
    return {image_name: gmm_params}

def process_folder(input_folder, output_folder):
    print(f"Processing images in folder: {input_folder}")
    all_gmm_params = {}
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image_gmm_params = process_image(image_path, output_folder)
            all_gmm_params.update(image_gmm_params)
    
    # Save all GMM parameters to a single JSON file
    params_output = os.path.join(output_folder, "all_gmm_params.json")
    with open(params_output, 'w') as f:
        json.dump(all_gmm_params, f, indent=2)
    print(f"All GMM parameters saved to {params_output}")

# Main execution
if __name__ == "__main__":
    print("Starting batch image processing...")
    input_folder = "../researchWork/Gaze_Enabled_Histopathology/SyntheticDataGeneration/data/masks/"  # Folder containing input images
    output_folder = "output"  # Folder for all output files and subfolders

    os.makedirs(output_folder, exist_ok=True)
    process_folder(input_folder, output_folder)

    print("Batch image processing finished successfully.")
