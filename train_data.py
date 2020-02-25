import sys
import numpy
import cv2
import os

# Image for the computer to detect contours and learn from
image_to_train = "training.png"

# Its 48 because that way the exact number of all letters are actually found
image_edges = 48

# Constant size variables when processing image and contours 
resized_image_w = 20
resized_image_h = 30


def train_data():

	# Index that will be associated with contour
	index_to_add = 0

	# Keeps track of sorted contour
	count = 0

	# Displays how many contours (letters) in training image it has found.
	# In the image provided, there are 26 x 22 letters, so it should find 572 letters to train from
	found_count = 0

	# Read in training images for all orientations
	training_image = cv2.imread(image_to_train)

	# Get grayscale and blur all images
	gray_image = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
	blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)

	# Filter image to black and white. Use gaussian filter, invert background with foreground.
	image_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

	# Make copy of filtered image
	image_threshold_copy = image_threshold.copy()

	# Now find the contours, use copy of threshold
	countours, hierarchy= cv2.findContours(image_threshold_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# Numpy array used later
	flattened_image = numpy.empty((0, resized_image_w * resized_image_h))

	# empty list where found letters will be classified
	classification = []

	# Alphabet unicodes that should be recognized
	all_chars = [ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
					 ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
					 ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

        # Sort all found contours from Top to Bottom
	sorted_ctrs = sorted(countours, key=lambda countours: cv2.boundingRect(countours)[0])

	# Loop through each contour detected
	for i in sorted_ctrs:

		# Check if its big enough
		if cv2.contourArea(i) > image_edges:

			found_count = found_count +1

			# Get bounding rect
			[intX, intY, intW, intH] = cv2.boundingRect(i)

			# Crop out the character
			cropped_character = image_threshold[intY:intY+intH, intX:intX+intW]

			# Resize for consistenty. Makes easier recognition and storage
			resize_cropped = cv2.resize(cropped_character, (resized_image_w, resized_image_h))

                        # There are 22 columns (22 variations of each letter in english alphabet) in the provided training image
			if count < 22:
				classification.append(all_chars[index_to_add])

			elif count == 22:
				count = 0
				classification.append(all_chars[index_to_add])
				index_to_add = index_to_add + 1

			# Flatten image to 1d numpy array and add to array for easy writing to .txt file
			flat_image = resize_cropped.reshape((1, resized_image_w * resized_image_h))
			flattened_image = numpy.append(flattened_image, flat_image, 0)
			count = count + 1


        # Convert classification array to floats and then reshape to 1d, so it can be written to text file
	float_classification = numpy.array(classification, numpy.float32)
	flat_classification = float_classification.reshape((float_classification.size, 1))

        # Save data in text files
	numpy.savetxt("classifications.txt", flat_classification)
	numpy.savetxt("flattened_images.txt", flattened_image)

        # General feedback
	print("Training completed!")
	print("Found contours: {}".format(found_count))

# Call main method
train_data()
