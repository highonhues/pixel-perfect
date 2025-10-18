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

def auto_detect_params(img):
    """Automatically detect the optimal params based on image stats.
    NOTE:Heuristic auto tuner based off of literature conventions to test if it works"""

    mean = np.mean(img)
    std = np.std(img)
    height, width = img.shape[:2]
    contrast =std / (mean + 1e-10)
    
    # CLAHE params
    if contrast < 0.3:
        clip_lmt = 3.0  #contrast how hence higher enhancement
    elif contrast < 0.5:
        clip_lmt = 2.5
    else:
        clip_lmt = 2.0
    
    tile_size = max(4, min(32, width // 32))  # scale with image size
    
    # Illum correction sigma
    sigma_illum = max(50, min(150, width // 10))
    
    # I came in like a Rolling ball 
    radius =max(20, min(100, width// 20))
    
    #gaussian sigma
    snr = mean / (std + 1e-10)
    if snr < 5:
        sigma_gauss = 1.5  # Noisy directly prop to smoothing
    elif snr < 10:
        sigma_gauss = 1.2
    else:
        sigma_gauss = 1.0

    #Median filter kernel
    if snr < 8:
        kernel_median = 5 
    else:
        kernel_median = 3
    
    return {
        'clahe_clip': clip_lmt,
        'clahe_tile': tile_size,
        'illum_sigma': sigma_illum,
        'rolling_radius': radius,
        'gauss_sigma' : sigma_gauss,
        'median_kernel' : kernel_median
    }
    

# Fin pipeline 
def preprocess_image(img, use_clahe=False, use_illum=False,use_rolling=False,
                     use_gauss=False, use_median=False, auto_params=True,params=None ):
    

    """Apply the whole preprocessing pipeline to your image ARGS AND OUTPUT FOR ALL TBD Ill write soon"""  
    
    result = img.copy()
    metrics_before = calculate_quality_metrics(img)

    # Get params
    if auto_params:
        use_params = auto_detect_params(img)
    else:
        use_params = params if params else {} 
    
    if use_clahe:
        result = clahe(result, 
                       clip_lmt=use_params.get('clahe_clip', 2.0), 
                       tile_size=use_params.get('clahe_tile', 8))
    
    if use_illum:
        result = correct_illumination(result, sigma=use_params.get('illum_sigma', 50))
    
    if use_rolling:
        result = rolling_ball_bg(result, radius=use_params.get('rolling_radius', 50))
    
    if use_gauss:
        result = gaussian_denoise(result, sigma=use_params.get('gauss_sigma', 1.0))
    
    if use_median:
        result = median_filter(result, kernel_size=use_params.get('median_kernel', 3))
    
    # Get metrics after processing
    metrics_after = calculate_quality_metrics(result)
    
    return result, metrics_before, metrics_after, use_params

#Real iamges
def load_image(filepath, return_colour_info=False):
    """
    Load an image file (supports TIFF, PNG, JPG)
    
    Args:
    filepath: Path to image file
    return_colour_info: If True returns image, is_colour, num_channels
        
    Returns: Image as numpy array
    """
    image =cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Could not load image: {filepath}")
    
    # Grayscale is 1 channel
    # Colour is 3 channels
    #Colour with alpha 4 channels
    is_colour = len(image.shape) == 3
    num_channels = image.shape[2] if is_colour else 1 
    
    print(f"Loaded: {filepath}")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Colour: {'Present' if is_colour else 'No'}")
    if is_colour:
        print(f"Channels: {num_channels}")
    
    if return_colour_info:
        return image, is_colour,num_channels
    return image


def process_channel(channel, use_clahe=False, use_illum=False, 
                    use_rolling=False, use_gauss=False, use_median=False,
                    auto_params=True, params=None):
    """
    Process a single channel (grayscale image) which is same as preprocess_image but for one channel only
    """
    return preprocess_image(channel, use_clahe, use_illum, use_rolling, 
                          use_gauss, use_median, auto_params,params)


def preprocess_colour_image(image, use_clahe=False, use_illum=False,
                           use_rolling=False, use_gauss=False, use_median=False,
                           auto_params=True, params=None):
    """
    Process a colour image by processing each channel separately
    
    Args:
        image: Input image (can be grayscale or colour)
        [other args same as preprocess_image]
        
    Returns:
        If grayscale: same as preprocess_image
        If color: (processed_color_image, metrics_before, metrics_after, used_params)
    """
    # If grayscale, use normal pipeline
    if len(image.shape) == 2:
        return preprocess_image(image, use_clahe, use_illum, use_rolling,
                               use_gauss, use_median, auto_params, params)
    
    # If colour, process each channel separately
    channels = cv2.split(image)
    processed_channels = []
    
    print(f"Processing {len(channels)} channels separately:-")
    
    for i, channel in enumerate(channels):
        print(f"Channel: {i+1}/{len(channels)}")
        processed, _, _, _ = preprocess_image(
            channel, use_clahe, use_illum, use_rolling,
            use_gauss, use_median, auto_params, params
        )
        processed_channels.append(processed)
    
    # Merge channels back
    result = cv2.merge(processed_channels)
    
    # Calculate metrics on first channel (or average)
    metrics_before = calculate_quality_metrics(channels[0])
    metrics_after = calculate_quality_metrics(processed_channels[0])
    
    used_params = auto_detect_params(channels[0]) if auto_params else (params or {})
    
    return result, metrics_before, metrics_after, used_params


def save_image(image, filepath):
    """
    Save an image to file.
    
    Args:
        image: Numpy array
        filepath: Output path
    """
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")




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

    # Auto Parameter Detection
    print("Testing Auto Param detection")
    auto_params = auto_detect_params(test_image)
    print("Auto-detected parameters:")
    for key, val in auto_params.items():
        print(f"  {key}: {val}")
    print()

    # Test Pipeline
    print("Testing Processing Pipeline...")
    processed, before, after, use_params = preprocess_image(
        test_image,use_clahe=True,use_illum=True,use_gauss=True,auto_params=True
    )
    cv2.imwrite(os.path.join(test_results_dir, 'test_pipeline.png'), processed)
    print("Saved test_pipeline.png")
    
    print("\nPipeline metrics before:")
    for key, val in before.items():
        print(f"{key}: {val:.4f}")
    
    print("\nPipeline metrics after:")
    for key, val in after.items():
        print(f"{key}: {val:.4f}")
    print()

    #colour Image Processing
    print("Testing colour img processing:-")
    colour_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_results_dir, 'test_color_input.png'), colour_image)
    
    processed_colour, _, _, _ = preprocess_colour_image(
        colour_image,
        use_clahe=True,
        use_gauss=True,
        auto_params=True
    )
    cv2.imwrite(os.path.join(test_results_dir, 'test_colour_output.png'), processed_colour)
    print("Saved test_colour_input.png and test_colour_output.png\n")

    print("All tests complete!")
    print(f"Check the {test_results_dir} folder for results")


if __name__ == "__main__":
    test_features()