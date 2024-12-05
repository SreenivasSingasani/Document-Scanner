# Document-Scanner
# importing the necessary libraries
from transform import four_point_transform
from skimage.filters import threshold_local

# this will help us to obtain black and white feel to our scanned image
import numpy as np
import argparse
import cv2
import imutils

# constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# Step 1: Edge Detection

# load the image and compute the ratio of the old height to the new height, clone it, resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
# ratio of original height to the new height
orig = image.copy()
image = imutils.resize(image, height=500)
# we are setting our scanned image to have a height of 500 pixels
# convert the image to grayscale and blur it and find the edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gaussian blur to remove high frequency noise

edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected one
print("STEP 1: Detect the Edges")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding Contours
# find the contours on the edged image, keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over all the contours
for c in cnts:
    # approx the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the approximated contour has 4 points, we found our screen
    if len(approx) == 4:
        screenCnt = approx
        break


# show the contour
print("STEP 2: Find Contours on paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply a perspective transform and threshold
# apply the four point transform to obtain top down view

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert it to grayscale, then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# show
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
# importing the necessary packages
# numpy for numerical processing
import numpy as np
import cv2


# we define a function order_points, that takes a single
# argument "pts", which is a list of four points specifying
# corners of a rectangle


def order_points(pts):
    # initialize a list of co od that will be ordered
    # such that the first entry in the list id the top-left
    # the second entry is top-right and the third is bottom right
    rect = np.zeros((4, 2), dtype="float32")
    # four orderes points

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
