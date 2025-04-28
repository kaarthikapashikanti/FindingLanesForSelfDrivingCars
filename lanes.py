import cv2
import numpy as np

# import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # (height,)
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Fit a first degree polynomial y = mx+c with points and return a vector which describes the slope and intercept
        # left side have negative slope
        # right side have positive slope
        parameters = np.polyfit(
            (x1, x2), (y1, y2), 1
        )  # 1 is degree output: [slope,y-intercept]
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # axis = 0 it gives the average of every column
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    # Converts one image to another color image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # it removes the noise by making it blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # it uses derivate and find the edges that are greater than thershold
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    # each line is a 2D array containing our line coordinates in the form [[x1,y1,x2,y2]]. These coordinates specify the lines parameters, as well as the location
    # of the lines with respect to the image space,ensuring that they are placed in the correct position
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # black image,first point,second point,color,thickness
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    # fill the mask dimension (black color and trace the triangle with color white(255))
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# For image
# Read the image and return the multi-dimensional numpy array containing the relative pixels
# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# #image,precision of 2 pixels and 1 degree precision, threshp;d(minimum number of votes needed to accept a candidate line)
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=50)
# avgeraged_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image,avgeraged_lines)
# #sum of our color image with our line_image
# #lane_image has a weight of 0.8 and line_image of weight 1 that means the line_image has more 20% so it will be darker and densed line compared to lane_image
# combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
# cv2.imshow("result",combo_image)
# #It displays the image (result) for given time 0 -> infinity
# cv2.waitKey(0)
# plt.show()


cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=50,
    )
    avgeraged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, avgeraged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    # mask the integer value from waitKey to eight bits
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
