import numpy as np
import cv2

def depthwise_convolution(image, kernel):

    num_channels = image.shape[2]
    output = np.zeros_like(image)
    
    for c in range(num_channels):
        output[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
    
    return output

def pointwise_convolution(image, kernel):

    if kernel.shape != (1, 1, image.shape[2]):
        raise ValueError("Kernel must have shape (1, 1, num_channels)")
    
    # Since it's a 1x1 convolution, we can use depthwise convolution function directly.
    return depthwise_convolution(image, kernel[:,:,0])

def apply_convolution(image_path, kernel, mode='depthwise'):
 
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    if mode == 'depthwise':
        convoluted_image = depthwise_convolution(image, kernel)
    elif mode == 'pointwise':
        convoluted_image = pointwise_convolution(image, kernel)
    else:
        raise ValueError("Invalid mode. Choose either 'depthwise' or 'pointwise'.")
    
    return convoluted_image
