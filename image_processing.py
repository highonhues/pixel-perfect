import numpy as np
import cv2
import os
from skimage import exposure

# Features

def gaussian_denoise(img, sigma=1.0):
    """ Use gaussian blur to smoothen image """
    return cv2.GaussianBlur(img,(0,0),sigma)


def clahe(img, clip_lmt=2.0, tile_size=8):
    """ Improve contrast using CLAHE Contrast Limited Adaptive Histogram .
     Adjust contrast in each part separately """
    clh = cv2.createCLAHE(clipLimit= clip_lmt, tileGridSize=(tile_size, tile_size))
    
    # 16 bit images convered to 8 bit and then back
    if img.dtype == np.uint16:

        img_8bit = (img /256).astype(np.uint8)
        result = clh.apply(img_8bit)
        
        return (result.astype(np.uint16) * 256)
    else:
        return clh.apply(img)


def median_filter(img, kernel_size=3):
    """Removed salt and pepepr noise"""

    # check kernel size to be odd
    if kernel_size % 2 ==0:
        kernel_size +=1
    return cv2.medianBlur(img, kernel_size)


def correct_illumination(img, sigma=50):
    """Fix uneven illumination by smoothening and obtaining the background and subtracting it."""
    
    # Estimate background with extreme blur
    background= cv2.GaussianBlur(img, (0,0), sigma)
    corrected= cv2.subtract(img, background)

    # Subtract background and normalise it
    if img.dtype == np.uint16:
        correctfin = cv2.normalize(corrected, None, 0, 65535, cv2.NORM_MINMAX)
    else:
        correctfin = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    return correctfin

def rolling_ball_bg(img, radius=50):
    """Rolling ball algortihm to estimate background intensity"""    
    # Create circular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    
    #morphological opening of rolling ball
    background= cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Subtract background
    return cv2.subtract(img, background)


# Qua;ity Metrics
def calculate_quality_metrics(image):
    """Calculate image quality metrics."""
    mean = np.mean(image)
    std = np.std(image)
    
    # rms contrast
    contrast = std/(mean + 1e-10)
    
    # SNR - Signal-to-Noise Ratio
    snr = mean/(std + 1e-10)
    
    #sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Dynamic Range
    min_val, max_val = image.min(), image.max()
    bit_depth = 65535 if image.dtype == np.uint16 else 255
    dynamic_range = (max_val - min_val) / bit_depth
    
    return {
        'contrast': float(contrast),
        'snr': float(snr),
        'sharpness': float(sharpness),
        'dynamic_range': float(dynamic_range)
    }


### test features


def test_features():
    """Test features and see the results"""

    base_dir= os.path.dirname(__file__)
    test_results_dir = os.path.join(base_dir, "test")
    os.makedirs(test_results_dir, exist_ok=True)
    
    # creating noisy low contrast image
    test_image = np.random.randint(50, 150, (512, 512), dtype=np.uint8)

    # Add gradient (uneven illumination)
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    gradient = (x / 512 * 100).astype(np.uint8)
    test_image = cv2.add(test_image, gradient)

    # Add some noise
    noise = np.random.randint(-30, 30, (512, 512))
    test_image = np.clip(test_image.astype(int) + noise, 0, 255).astype(np.uint8)
   
    #save pic
    cv2.imwrite(os.path.join(test_results_dir, 'test_input.png'), test_image)
    print(f" Test image of {test_image.shape} created")

    #Guassian Denoise
    try:
        denoised = gaussian_denoise(test_image, sigma=1.5)
        cv2.imwrite(os.path.join(test_results_dir,'test_gaussian.png'), denoised)
        print("Guassian denoise ran check test_gaussian.png")
    except Exception as e:
        print(f"Error {e}")
    
    #CLAHE
    try:
        bettercon = clahe(test_image)
        cv2.imwrite(os.path.join(test_results_dir,'test_clahe.png'), bettercon)
        print("CLAHE ran check test_clahe.png")
    except Exception as e:
        print(f"Error {e}")

    #median
    try:
        filtered = median_filter(test_image)
        cv2.imwrite(os.path.join(test_results_dir,'test_median.png'), filtered)
        print("median_filter ran check test_median.png")
    except Exception as e:
        print(f"Error {e}")

    # Rolling dat ball
    try:
        rbi = rolling_ball_bg(test_image)
        cv2.imwrite(os.path.join(test_results_dir,'test_rollingball.png'), rbi)
        print("rolling_ball_bg ran check test_rollingball.png")
    except Exception as e:
        print(f"Error {e}")


    #illumination
    try:
        illcorr = correct_illumination(test_image)
        cv2.imwrite(os.path.join(test_results_dir,'test_illum.png'), illcorr)
        print("correct_illumination ran check test_illum.png")
    except Exception as e:
        print(f"Error {e}")

    #Check metrics
    print("Testing Quality Metrics:\n")
    original_metrics = calculate_quality_metrics(test_image)
    processed_metrics = calculate_quality_metrics(illcorr)
    Rolling_metrics = calculate_quality_metrics(rbi)

    print("Metrics of OG LQ picture:\n")
    for key, val in original_metrics.items():
        print(f"     {key}: {val:.4f}")

    print("Metrics of illum pics:\n")
    for key, val in processed_metrics.items():
        print(f"     {key}: {val:.4f}")
    
    print("Metrics of Rolling ball algo pics:\n")
    for key, val in Rolling_metrics.items():
        print(f"     {key}: {val:.4f}")


if __name__ == "__main__":
    test_features()