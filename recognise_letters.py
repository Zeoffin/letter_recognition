import cv2
import numpy
import operator
import os

# Define
image_edges = 48

resized_image_w = 20
resized_image_h = 30

image_to_recognise = "test_images/alphabet.png"


# Class for detecting each individual letter in a provided image
class ContourWithData:

    # Variables
    contour = None
    bounding_rect = None
    rectangle_x = 0
    rectangle_y = 0
    rectangle_w = 0
    rectangle_h = 0
    area = 0.0

    def calculate_bounds(self):

        [int_x, int_y, int_width, int_height] = self.bounding_rect
        self.rectangle_x = int_x
        self.rectangle_y = int_y
        self.rectangle_w = int_width
        self.rectangle_h = int_height

    # TODO: Make this better to validate
    def contour_valid(self):

        if self.area < image_edges:
            return False
        else:
            return True


def main():

    all_contours_with_data = []
    valid_contours_with_data = []

    # read in training classifications
    classifications = numpy.loadtxt("classifications.txt", numpy.float32)

    # read in training images
    flattened_images = numpy.loadtxt("flattened_images.txt", numpy.float32)

    # reshape numpy array to 1d array
    classifications = classifications.reshape((classifications.size, 1))

    # Initiate knn
    k_nearest = cv2.ml.KNearest_create()
    k_nearest.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)

    # read image in which letters will be recognised
    letter_image = cv2.imread(image_to_recognise)

    # Set grayscale and blur the image with letter in it
    gray_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # filter image from grayscale to black and white
    image_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                            11, 2)


    # TODO: change SIMPLE --> NONE
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:

        # Create object - necessary for recognizing multiple letters in an image
        contour_with_data = ContourWithData()
        contour_with_data.contour = c
        contour_with_data.bounding_rect = cv2.boundingRect(contour_with_data.contour)
        contour_with_data.calculate_bounds()
        contour_with_data.area = cv2.contourArea(contour_with_data.contour)

        # Add to all contours
        all_contours_with_data.append(contour_with_data)

    for contour_with_data in all_contours_with_data:

        # Check for contour validity
        if contour_with_data.contour_valid():

            # If valid, add to valid array
            valid_contours_with_data.append(contour_with_data)

    # Sort from left to right
    valid_contours_with_data.sort(key=operator.attrgetter("rectangle_x"))

    # String for holding all detected letters
    detected_letters = ""

    for contour_with_data in valid_contours_with_data:

        # Crop out the letter detected
        cropped_letter = image_threshold[contour_with_data.rectangle_y: contour_with_data.rectangle_y + contour_with_data.rectangle_h,
                 contour_with_data.rectangle_x: contour_with_data.rectangle_x + contour_with_data.rectangle_w]

        # resize for convenience
        resized_letter = cv2.resize(cropped_letter, (resized_image_w, resized_image_h))

        # make it even more convenient in 1d floats
        flat_letter = resized_letter.reshape((1, resized_image_w * resized_image_h))
        flat_letter = numpy.float32(flat_letter)

        # Find the nearest k
        value, results, neigh_resp, dists = k_nearest.findNearest(flat_letter, k=1)

        # Get the determined letter and append to string which will be returned
        current_letter = str(chr(int(results[0][0])))
        detected_letters = detected_letters + current_letter

    # Print detected letters
    print(detected_letters)


main()





