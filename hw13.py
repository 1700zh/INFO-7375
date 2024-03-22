import numpy as np

def convolution_2d(image, filter):
    # Dimensions of the image
    image_height, image_width = image.shape
    
    # Dimensions of the filter
    filter_height, filter_width = filter.shape
    
    # Calculate the dimensions of the output image
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    
    # Iterate over every possible position to apply the filter
    for i in range(output_height):
        for j in range(output_width):
            # Perform element-wise multiplication and sum up the results
            output[i, j] = np.sum(image[i:i+filter_height, j:j+filter_width] * filter)
            
    return output

# Example
image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

filter = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

output = convolution_2d(image, filter)
print("Convolved feature map:\n", output)
