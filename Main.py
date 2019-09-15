import cv2
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Three_Functions import histogram
from Three_Functions import histSpecification
from Three_Functions import LaplacianFilter
from Three_Functions import fullScaleContrastStretch

def display_image(window_name, image):
    """A function to display image"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     interpolation method and writes the output image"""

    #Parse input arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the input image", metavar="IMAGE")
    parser.add_argument("-d", "--desired_image", dest="image2",
                        help="specify the name of the desired image", metavar="IMAGE")

    args = parser.parse_args()

    #Load image
    if args.image is None:
        print("Please specify the name of image")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        image_name = args.image.split(".")[0]
        input_image = cv2.imread(args.image, 0)


    if args.image2 is None:
        print("Please specify the name of image")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        image_name = args.image.split(".")[0]
        desired_image = cv2.imread(args.image2, 0)

    #  compute cumulative histogram


    outputDir = 'output/'
    histShaping = histSpecification(input_image, desired_image)
    output_shaping_name = outputDir + "histogram_shaping_image_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_shaping_name, histShaping)

    #  contrast stretch
    contrast_stretch = fullScaleContrastStretch(input_image)
    output_image_name = outputDir + "contrast_stretch_image_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_image_name, contrast_stretch)

    #  Laplacian filter(sharpening)
    laplacian_filter = LaplacianFilter(input_image)
    output_laplacian_name = outputDir + "Laplacian_filtered_image_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_laplacian_name, laplacian_filter)


if __name__ == "__main__":
    main()