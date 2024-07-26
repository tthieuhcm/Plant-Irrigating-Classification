import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2


def flip_image(image, horizontal=True):
    if horizontal:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_image(image, degrees=180):
    return image.rotate(degrees)

def add_noise(image):
    np_image = np.array(image)
    # Generate noise
    noise = np.random.normal(scale=25, size=np_image.shape)
    noisy_image = np_image + noise
    # Ensure we don't go out of bounds in pixel values
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(np.uint8(noisy_image))

def change_contrast(image, level=2):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(level)

def change_brightness(image, level=1.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(level)

def split_image_into_parts(image, parts=4):
    width, height = image.size
    mid_width, mid_height = width // 2, height // 2
    
    # Define the bounding boxes for the four parts
    top_left = (0, 0, mid_width, mid_height)
    top_right = (mid_width, 0, width, mid_height)
    bottom_left = (0, mid_height, mid_width, height)
    bottom_right = (mid_width, mid_height, width, height)
    
    # Crop the image into four parts
    image_top_left = image.crop(top_left)
    image_top_right = image.crop(top_right)
    image_bottom_left = image.crop(bottom_left)
    image_bottom_right = image.crop(bottom_right)
    
    return image_top_left, image_top_right, image_bottom_left, image_bottom_right

# Example usage
def process_and_save_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            if '215222077834' in filename:
                treatment = 0
            else:
                treatment = 1
            image = Image.open(os.path.join(input_folder, filename))
            # Split the image
            parts = split_image_into_parts(image)
            for i in range(4):
                if treatment == 0:
                    label = 0 if i in [0, 1] else 1
                else: 
                    label = 3 if i in [0, 1] else 2
                # # Apply transformations
                # flipped_horizontally = flip_image(parts[i])
                # flipped_vertically = flip_image(parts[i], horizontal=False)
                # rotated = rotate_image(parts[i])
                # noised = add_noise(parts[i])
                # higher_contrast = change_contrast(parts[i])
                # brighter_image = change_brightness(parts[i])

                # # Save the transformed images
                # flipped_horizontally.save(os.path.join(output_folder, f'{label}_{filename}_{i}_flipped_horizontal.png'))
                # flipped_vertically.save(os.path.join(output_folder, f'{label}_{filename}_{i}_flipped_vertical.png'))
                # rotated.save(os.path.join(output_folder, f'{label}_{filename}_{i}_rotated_180.png'))
                # noised.save(os.path.join(output_folder, f'{label}_{filename}_{i}_noised.png'))
                # higher_contrast.save(os.path.join(output_folder, f'{label}_{filename}_{i}_higher_contrast.png'))
                # brighter_image.save(os.path.join(output_folder, f'{label}_{filename}_{i}_brighter.png'))
                parts[i].save(os.path.join(output_folder, f'{label}_{filename}_{i}.png'))
                parts[i].save(os.path.join(output_folder, f'{label}_{filename}_{i}.png'))
                parts[i].save(os.path.join(output_folder, f'{label}_{filename}_{i}.png'))
                parts[i].save(os.path.join(output_folder, f'{label}_{filename}_{i}.png'))


def is_too_black(image, darkness_threshold=60, black_percentage_threshold=50):
    """
    Determine if an image is too black.
    :param image: PIL Image object.
    :param darkness_threshold: Pixel value below which is considered dark (0-255).
    :param black_percentage_threshold: Percentage of dark pixels above which the image is too black.
    :return: Boolean indicating if the image is too black.
    """
    grayscale = image.convert("L")  # Convert to grayscale for brightness analysis
    total_pixels = grayscale.size[0] * grayscale.size[1]
    dark_pixels = len([pixel for pixel in grayscale.getdata() if pixel < darkness_threshold])
    dark_percentage = (dark_pixels / total_pixels) * 100

    return dark_percentage > black_percentage_threshold

# Example usage
if __name__ == "__main__":
    input_folder = 'C:/Users/PhongTran/PycharmProjects/RealSenseTest/original_dataset'
    # for filename in os.listdir(input_folder):
    #     if filename.endswith(".png"):
    #         image = Image.open(os.path.join(input_folder, filename))
    #         if is_too_black(image):
    #             os.remove(os.path.join(input_folder, filename))
    #             print(f"Removed {filename} because it was too black.")
    # print("Done!!!")
    # input_folder = 'C:/Users/PhongTran/PycharmProjects/RealSenseTest/original_dataset'
    output_folder = 'C:/Users/PhongTran/PycharmProjects/RealSenseTest/modified_dataset'
    process_and_save_images(input_folder, output_folder)