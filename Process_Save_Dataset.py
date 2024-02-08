# Import necessary libraries
import os
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

def preprocess_image(image_path, target_size=(380, 380)):
    original_image = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(original_image, threshold, 1, cv2.INPAINT_TELEA)
    resized_image = cv2.resize(inpainted_image, target_size)
    resized_image_bgr = resized_image
    return resized_image_bgr

def color_constancy(img, power=6, gamma=2.2):
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256): look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)

# Load data with tqdm progress bar
mel_folder = 'C:/Users/aacer/PycharmProjects/TP1/PFEM2IMOVI/Dataset/MEL'
nev_folder = 'C:/Users/aacer/PycharmProjects/TP1/PFEM2IMOVI/Dataset/NEV'

mel_images_all = []
nev_images_all = []

# Load and preprocess melanoma images
print("Loading and Preprocessing Melanoma Images:")
for filename in tqdm(os.listdir(mel_folder), desc="Melanoma"):
    mel_image = preprocess_image(os.path.join(mel_folder, filename))
    mel_images_all.append(color_constancy(mel_image))

# Load and preprocess nevus images
print("Loading and Preprocessing Nevus Images:")
for filename in tqdm(os.listdir(nev_folder), desc="Nevus"):
    nev_image = preprocess_image(os.path.join(nev_folder, filename))
    nev_images_all.append(color_constancy(nev_image))



# Create folders to save processed images
save_folder_mel = '/kaggle/working/Processed/MEL'
save_folder_nev = '/kaggle/working/Processed/NEV'

os.makedirs(save_folder_mel, exist_ok=True)
os.makedirs(save_folder_nev, exist_ok=True)

# Save processed melanoma images
print("Saving Processed Melanoma Images:")
for i, mel_image in enumerate(tqdm(mel_images_all, desc="Melanoma")):
    save_path = os.path.join(save_folder_mel, f"processed_mel_{i}.jpg")
    cv2.imwrite(save_path, mel_image)

# Save processed nevus images
print("Saving Processed Nevus Images:")
for i, nev_image in enumerate(tqdm(nev_images_all, desc="Nevus")):
    save_path = os.path.join(save_folder_nev, f"processed_nev_{i}.jpg")
    cv2.imwrite(save_path, nev_image)

