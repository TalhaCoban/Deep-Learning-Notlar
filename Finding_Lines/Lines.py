import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #blurring: reduces noises
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #derivative(f(x,y)): measures adjacent changes in intensity in all directions, x and y
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [[200, height], [1100, height], [550,250]]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


image = cv2.imread("test_image.jpg")
#Edge Detection: identifiying sharp changes in intesnity in adjacent pixels
#gradient : measure of change in brigthness over adjacent pixels
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
#Line of Lane
#cv2.HoughLinesP(image, rho_size(pixel), angel_size, threshold, placeholder array, lenght of a line in pixels that we will accept into the output,)
#threshold : to find and display the line from a series of dots. threshold is the minimum nunmber of intersections needed to detect the line
#What is the optimal number of votes where we can say Ok draw the line that corresponds to the spin
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(combo_image, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("result", resized)
cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    #Line of Lane
    #cv2.HoughLinesP(image, rho_size(pixel), angel_size, threshold, placeholder array, lenght of a line in pixels that we will accept into the output,)
    #threshold : to find and display the line from a series of dots. threshold is the minimum nunmber of intersections needed to detect the line
    #What is the optimal number of votes where we can say Ok draw the line that corresponds to the spin
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

