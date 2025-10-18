import numpy as np
import cv2
import os
from skimage import exposure

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




# test features


def test_features():
    """Test features and see the results"""

    base_dir= os.path.dirname(__file__)
    test_results_dir = os.path.join(base_dir, "test")
    os.makedirs(test_results_dir, exist_ok=True)
    
    # creating noisy low contrast image
    test_image = np.random.randint(50, 150, (512, 512), dtype=np.uint8)
    # Add some noise
    noise = np.random.randint(-30, 30, (512, 512))
    test_image = np.clip(test_image.astype(int) + noise, 0, 255).astype(np.uint8)
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



if __name__ == "__main__":
    test_features()