import cv2
import numpy
import operator
import os
import statistics
from PIL import Image

# Define
image_edges = 48

resized_image_w = 20
resized_image_h = 30

# K number used in k-nearest neighbor. Odd, so there are no ties when decideing
k_number = 23

image_to_recognise = "test_images/z.png"


# Class for saving data for detected letter
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

    def contour_valid(self):

        if self.area < image_edges:
            return False
        else:
            return True


def rotate_image(image_to_rotate, angle):
    rotated_image = cv2.imread(image_to_rotate)

    (h, w) = rotated_image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
 
    scale = 1.0
 
    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated90 = cv2.warpAffine(rotated_image, M, (h, w))  

    return rotated90


def main():

    print("Detecting letter \n")

    all_detected_letters = []
    all_letter_avarage_distance = []

    angle = 0

    # Check for all possible 4 rotations
    for n in range(4):
        all_contours_with_data = []
        valid_contours_with_data = []

        # read in training classifications
        classifications = numpy.loadtxt("classifications.txt", numpy.float32)

        # read in training images
        flattened_images = numpy.loadtxt("flattened_images.txt", numpy.float32)

        # reshape numpy array to 1d array
        classifications = classifications.reshape((classifications.size, 1))

        # Initiate k-nearest neighbor
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)

        # read image in which letters will be recognised
        if angle == 0:
            letter_image = cv2.imread(image_to_recognise)
        else:
            letter_image = rotate_image(image_to_recognise, angle)

        # Set grayscale and blur the image with letter in it
        gray_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # filter image from grayscale to black and white
        image_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                11, 2)


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
            value, results, neighbours, distance = k_nearest.findNearest(flat_letter, k=k_number)

            # Get the determined letter and append to string which will be returned
            current_letter = str(chr(int(results[0][0])))
            detected_letters = detected_letters + current_letter

        avg_dis_list = []
        # Calculate the avarage distance for each rotation of image
        if valid_contours_with_data[-1]:

            distance_instances = 0
            avg_dis_sum = 0

            for i in distance:
                for x in i:
                    distance_instances = distance_instances + 1
                    avg_dis_sum = avg_dis_sum + x

            avg_dis = avg_dis_sum/distance_instances

            avg_dis_list.append(avg_dis)


            # Appends all avarage distances and all letters of those distances
            all_detected_letters.append(detected_letters[-1])
            all_letter_avarage_distance.append(avg_dis_list[-1])
        
        angle = angle + 90

    # String which will hold detected letter
    real_letter = ""

    # A big number for convenience
    shortest_distance = 50000000

    # Find the shortest distance
    for i in all_letter_avarage_distance:
        x = x + i
        if i < shortest_distance:
            shortest_distance = i 

    # Take the letter, which has shortes distance from list. The index remains the same
    real_letter = all_detected_letters[all_letter_avarage_distance.index(shortest_distance)]

    print("All possible letters found through rotations: ")
    print(all_detected_letters)
    print("\n")
    print("All distances correspoding to each found letter: ")
    print(all_letter_avarage_distance)
    print("\n")
    print("Determined letter: {}".format(real_letter))

# Start the program
main()





