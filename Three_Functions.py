import cv2
import numpy as np

def histogram(image):
    height, width = image.shape
    originalHist = np.zeros(256, dtype=np.float32)
    cumulativeHist = np.zeros(256, dtype=np.float32)

    for h in range(height):
        for w in range(width):
            originalHist[image[h][w]] += 1

    cumulativeHist[0] = originalHist[0]

    for h in range(1, 256):
        cumulativeHist[h] = cumulativeHist[h - 1] + originalHist[h]
    result = cumulativeHist / (height * width)

    return result


def histSpecification(input_image, desired_image):

    input_image_hist = histogram(input_image)
    desired_image_hist = histogram(desired_image)
    value = np.zeros(256, dtype=np.uint8)

    for h in range(256):
        diff = np.abs(input_image_hist[h] - desired_image_hist[h])
        matchValue = h
        for w in range(256):
            if np.abs(input_image_hist[h] - desired_image_hist[w]) < diff:
                diff = np.abs(input_image_hist[h] - desired_image_hist[w])
                matchValue = w
        value[h] = matchValue
    output = cv2.LUT(input_image, value)

    return output


def LaplacianFilter(image):

    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    laplacian = np.array([
                        [0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            value = laplacian * image[(h - 1):(h + 2), (w - 1):(w + 2)]
            result[h, w] = min(255, max(0, value.sum()))

    return result


def fullScaleContrastStretch(image):

    height, width = image.shape
    result_image = np.zeros((height, width), dtype=np.uint8)

    a = np.min(image)
    b = np.max(image)

    for h in range(height):
        for w in range(width):
            result_image[h][w] = (image[h][w] - a) / (b - a) * 255

    return result_image

