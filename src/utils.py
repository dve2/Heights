from scipy import ndimage
import numpy as np



def get_max_inside_blobs(image, mask):
    labels, num_labels = ndimage.label(mask)

    # Get max for each blob (indices 1 to num_labels inclusive)
    max_values = ndimage.maximum(image, labels=labels, index=np.arange(1, num_labels + 1))
    #Image.fromarray(image).save("image.png")
    max_points_2d = np.zeros_like(image)

    # Populate the 2D array with max values for each blob
    for label in range(1, num_labels + 1):
        y, x = np.where(labels == label)  # Coordinates of the blob
        if len(y) == 0:
            continue
        max_idx = np.argmax(image[y, x])  # Index of the max value within the blob
        y_max = y[max_idx]
        x_max = x[max_idx]
        max_points_2d[y_max, x_max] = image[y_max, x_max]
    return max_points_2d