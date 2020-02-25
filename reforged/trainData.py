import sys
import numpy as numpy
import cv2
import os

image_to_train = "letters.png"

image_edges = 100

resized_image_w = 20
resized_image_h = 30


def train_data():
	
	count = 0
	found_count = 0

	# Read in training images for all orientations
	training_image = cv2.imread(image_to_train)

	# Get grayscale and blur all images
	gray_image = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
	blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)

	# Filter image to black and white. Use gaussian filter, invert background with foreground.
	image_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
													cv2.THRESH_BINARY_INV, 11, 1)

	# Make copy of filtered image
	image_threshold_copy = image_threshold.copy()

	# Now find the contours, use copy of threshold
	countours, hierarchy= cv2.findContours(image_threshold_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Numpy array used later
	flattened_image = numpy.empty((0, resized_image_w * resized_image_h))

	# empty list where characters will be classified
	#classification = []

	# ALl possible characters that should be recognized
	all_chars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

	print("Training in progress. Please wait!")

	# Loop through each contour detected
	for i in countours:

		count = count +1

		# Check if its big enough
		if cv2.contourArea(i) > image_edges:
			
			found_count = found_count + 1

			# Get bounding rect
			[intX, intY, intW, intH] = cv2.boundingRect(i)

			# Crop out the character
			cropped_character = image_threshold[intY:intY+intH, intX:intX+intW]

			# Resize for consistenty. Makes easier recognition and storage
			resize_cropped = cv2.resize(cropped_character, (resized_image_w, resized_image_h))

			# Flatten image to 1d numpy array and add to array
			flat_image = resize_cropped.reshape((1, resized_image_w * resized_image_h))
			flattened_image = numpy.append(flattened_image, flat_image, 0)

	float_all_chars = numpy.array(all_chars, numpy.float32)

	flat_all_chars = float_all_chars.reshape((float_all_chars.size, 1))

	numpy.savetxt("classifications.txt", flat_all_chars)
	numpy.savetxt("flattened_images.txt", flattened_image)

	print("Training completed!\n")
	print(count)
	print("\n")
	print(found_count)

train_data()