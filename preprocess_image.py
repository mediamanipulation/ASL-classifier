"""
Image Preprocessing Pipeline for Better Real-World Performance
Helps normalize external images to match training data characteristics
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def preprocess_for_prediction(image_path, output_path=None):
    """
    Preprocess external images to improve prediction accuracy

    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save preprocessed image

    Returns:
        PIL.Image: Preprocessed image
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if cv2 fails
            img = Image.open(image_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = image_path

    print("Original shape:", img.shape)

    # Step 1: Remove background (make it more uniform)
    img_processed = remove_background(img)

    # Step 2: Improve contrast
    img_processed = improve_contrast(img_processed)

    # Step 3: Normalize brightness
    img_processed = normalize_brightness(img_processed)

    # Step 4: Resize to square with padding (preserve aspect ratio)
    img_processed = resize_with_padding(img_processed, target_size=128)

    # Convert to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))

    if output_path:
        img_pil.save(output_path)
        print(f"Preprocessed image saved to: {output_path}")

    return img_pil


def remove_background(img):
    """Remove or blur background to focus on hand"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to separate hand from background
    # Assuming hand is lighter than background (adjust if needed)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest contour (likely the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Dilate mask slightly to include hand edges
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Apply mask to original image
        result = cv2.bitwise_and(img, img, mask=mask)

        # Make background white instead of black
        background = np.full_like(img, 255)
        background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, background)

        return result

    return img


def improve_contrast(img):
    """Improve image contrast"""
    # Convert to PIL for easier contrast adjustment
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%

    # Convert back to cv2 format
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)


def normalize_brightness(img):
    """Normalize image brightness"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge([l, a, b])

    # Convert back to BGR
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_with_padding(img, target_size=128):
    """
    Resize image to target size while preserving aspect ratio
    Adds white padding to make it square
    """
    h, w = img.shape[:2]

    # Calculate scaling factor
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Calculate position to paste resized image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    # Paste resized image onto canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def batch_preprocess(input_dir, output_dir):
    """
    Preprocess all images in a directory

    Args:
        input_dir (str): Directory containing images to preprocess
        output_dir (str): Directory to save preprocessed images
    """
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    for filename in os.listdir(input_dir):
        if Path(filename).suffix.lower() in image_extensions:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"Processing: {filename}")
                preprocess_for_prediction(input_path, output_path)
                print(f"  -> Saved to: {output_path}")
            except Exception as e:
                print(f"  -> Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python preprocess_image.py input.jpg [output.jpg]")
        print("  Batch:        python preprocess_image.py input_dir/ output_dir/")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        import os
        if os.path.isdir(input_path):
            # Batch processing
            if not output_path:
                output_path = input_path + "_preprocessed"
            batch_preprocess(input_path, output_path)
        else:
            # Single image
            if not output_path:
                output_path = "preprocessed_" + os.path.basename(input_path)
            preprocess_for_prediction(input_path, output_path)
