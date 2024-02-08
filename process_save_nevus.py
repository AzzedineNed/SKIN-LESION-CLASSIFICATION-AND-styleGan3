import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path, target_size=(256, 256)):
    original_image = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(original_image, threshold, 1, cv2.INPAINT_TELEA)
    resized_image = cv2.resize(inpainted_image, target_size)
    resized_image_bgr = resized_image
    return resized_image_bgr

# Load data with tqdm progress bar
nev_folder = 'C:/Users/aacer/PycharmProjects/TP1/PFEM2IMOVI/Dataset/NEV'
nev_images_all = []

# Load and preprocess nevus images
print("Loading and Preprocessing Nevus Images:")
for filename in tqdm(os.listdir(nev_folder), desc="Nevus"):
    nev_image = preprocess_image(os.path.join(nev_folder, filename))
    nev_images_all.append(nev_image)

# Create folders to save processed images
save_folder_nev = '/kaggle/working/Processed256x256/NEV'
os.makedirs(save_folder_nev, exist_ok=True)

# Save processed nevus images
print("Saving Processed Nevus Images:")
for i, nev_image in enumerate(tqdm(nev_images_all, desc="Nevus")):
    save_path = os.path.join(save_folder_nev, f"processed_nev_{i}.jpg")
    cv2.imwrite(save_path, nev_image)
