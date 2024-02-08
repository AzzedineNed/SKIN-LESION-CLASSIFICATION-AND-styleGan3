import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # Optional, for displaying images

def preprocess_image(image_path, target_size=(380, 380)):
    # Load the image
    original_image = cv2.imread(image_path)

    # Resize the original image to the target size
    resized_original = cv2.resize(original_image, target_size)

    # Convert the resized image to grayscale
    gray_scale = cv2.cvtColor(resized_original, cv2.COLOR_BGR2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)

    # Intensify the hair contours in preparation for inpainting
    ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the resized original image depending on the mask
    inpainted_image = cv2.inpaint(resized_original, threshold, 1, cv2.INPAINT_TELEA)

    # Convert the inpainted image back to BGR color space
    resized_image_bgr = inpainted_image  # No need for cv2.cvtColor here

    return resized_original, resized_image_bgr

def color_constancy(img, power=6, gamma=2.2):
    """
    Parameters
    ----------
    img: 3D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in the reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in the reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)

# Directory path
directory_path = "C:/Users/aacer/PycharmProjects/TP1/PFEM2IMOVI/Dataset/NEV"

# Get the list of files in the directory
files = os.listdir(directory_path)

# Select the first 12 image files (you can modify this based on your needs)
selected_files = files[:12]

# Create a 4x3 subplot grid
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

# Iterate over the selected files
for i, file_name in enumerate(selected_files):
    # Preprocess the image
    image_path = os.path.join(directory_path, file_name)
    original_img, preprocessed_img = preprocess_image(image_path)

    # Apply color constancy
    result_img = color_constancy(preprocessed_img)

    # Display the original, preprocessed, and result images in the subplot
    axes[i // 3, i % 3].imshow(np.hstack([cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                                          cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB),
                                          cv2.cvtColor(result_img.astype('uint8'), cv2.COLOR_BGR2RGB)]))
    axes[i // 3, i % 3].set_title(f'Image {i + 1}')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()
