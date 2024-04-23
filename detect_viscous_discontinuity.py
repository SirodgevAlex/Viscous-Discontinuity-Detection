import cv2
import numpy as np

def find_viscous_discontinuity(image):
    # Convert the image to grayscale and enhance the contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, enhanced = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological closing operation to close small gaps between structures
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Apply morphological opening operation to separate structures
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)
    
    return opened

def color_viscous_discontinuity(image, mask):
    # Define the color for the viscous discontinuity area
    color = (0, 255, 0)  # Green color
    
    # Make a copy of the original image
    result = image.copy()
    
    # Find contours in the mask and draw them on the result image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness=1)
    
    return result

# Load the microscope image in JPEG format
image = cv2.imread('microscope_image.jpg')

# Find the viscous discontinuity area
viscous_discontinuity_mask = find_viscous_discontinuity(image)

# Color the viscous discontinuity area with lines
result_image = color_viscous_discontinuity(image, viscous_discontinuity_mask)

# Display and save the result
cv2.imshow('Result', result_image)
cv2.imwrite('result_image.jpg', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
