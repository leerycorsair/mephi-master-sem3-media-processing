import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

def calculate_mass(binary_img: np.ndarray) -> Tuple[List[int], List[float]]:
    height, width = binary_img.shape
    quarters = [
        binary_img[:height//2, :width//2],
        binary_img[:height//2, width//2:],
        binary_img[height//2:, :width//2],
        binary_img[height//2:, width//2:]
    ]
    masses = [np.sum(quarter == 255) for quarter in quarters]
    specific_weights = [mass / quarter.size for mass, quarter in zip(masses, quarters)]
    return masses, specific_weights

def calculate_center_of_gravity(binary_img: np.ndarray) -> Tuple[List[float], List[float]]:
    coords = np.argwhere(binary_img == 255)
    if len(coords) > 0:
        center_of_gravity = np.mean(coords, axis=0).tolist()
        normalized_cog = (center_of_gravity / np.array(binary_img.shape)).tolist()
    else:
        center_of_gravity, normalized_cog = [0, 0], [0, 0]
    return center_of_gravity, normalized_cog

def calculate_inertia(binary_img: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = np.argwhere(binary_img == 255)
    height, width = binary_img.shape
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    
    inertia_x = np.sum((y_coords - height / 2) ** 2) if len(coords) > 0 else 0
    inertia_y = np.sum((x_coords - width / 2) ** 2) if len(coords) > 0 else 0
    normalized_inertia_x = inertia_x / len(coords) if len(coords) > 0 else 0
    normalized_inertia_y = inertia_y / len(coords) if len(coords) > 0 else 0
    
    return (inertia_x, inertia_y), (normalized_inertia_x, normalized_inertia_y)

def calculate_profiles(binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_profile = np.sum(binary_img == 255, axis=0)
    y_profile = np.sum(binary_img == 255, axis=1)
    return x_profile, y_profile

def save_profile(profile: np.ndarray, filename: str, xlabel: str) -> None:
    plt.bar(range(len(profile)), profile)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(filename)
    plt.close()

def process_image(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    masses, specific_weights = calculate_mass(binary_img)
    center_of_gravity, normalized_cog = calculate_center_of_gravity(binary_img)
    moments_of_inertia, normalized_moments = calculate_inertia(binary_img)
    x_profile, y_profile = calculate_profiles(binary_img)
    
    return {
        "masses": masses,
        "specific_weights": specific_weights,
        "center_of_gravity": center_of_gravity,
        "normalized_cog": normalized_cog,
        "moments_of_inertia": moments_of_inertia,
        "normalized_moments": normalized_moments,
        "x_profile": x_profile,
        "y_profile": y_profile
    }

def process_directory(root_dir: str, output_dir: str, output_csv: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_data = []
    
    folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    
    for folder_name in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(root_dir, folder_name)
        styles = [style for style in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, style))]
        
        for style in tqdm(styles, desc=f"Processing styles in {folder_name}", leave=False):
            style_path = os.path.join(folder_path, style)
            images = [img for img in os.listdir(style_path) if img.endswith((".png", ".jpg"))]
            
            for img_file in tqdm(images, desc=f"Processing images in {style}", leave=False):
                img_path = os.path.join(style_path, img_file)
                result = process_image(img_path)
                
                csv_row = [
                    folder_name, style, img_file,
                    *result["masses"],
                    *result["specific_weights"],
                    *result["center_of_gravity"],
                    *result["normalized_cog"],
                    *result["moments_of_inertia"],
                    *result["normalized_moments"]
                ]
                csv_data.append(csv_row)
                
                x_profile_filename = os.path.join(output_dir, f"{folder_name}_{style}_{img_file}_x_profile.png")
                y_profile_filename = os.path.join(output_dir, f"{folder_name}_{style}_{img_file}_y_profile.png")
                save_profile(result["x_profile"], x_profile_filename, "X-axis")
                save_profile(result["y_profile"], y_profile_filename, "Y-axis")

    save_csv_data(csv_data, output_dir, output_csv)

def save_csv_data(csv_data: List[List[Any]], output_dir: str, output_csv: str) -> None:
    output_csv_path = os.path.join(output_dir, output_csv)
    header = [
        "Folder", "Style", "Image", "Mass Q1", "Mass Q2", "Mass Q3", "Mass Q4",
        "Specific Weight Q1", "Specific Weight Q2", "Specific Weight Q3", "Specific Weight Q4",
        "COG X", "COG Y", "Normalized COG X", "Normalized COG Y",
        "Inertia X", "Inertia Y", "Normalized Inertia X", "Normalized Inertia Y"
    ]

    with open(output_csv_path, mode="w", newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(header)
        writer.writerows(csv_data)

root_directory = "letter_images"
output_directory = "output"
output_csv_file = "features_output.csv"

process_directory(root_directory, output_directory, output_csv_file)
