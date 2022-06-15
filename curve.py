import time
import cv2
import sys
import numpy as np

video = cv2.VideoCapture("test.mp4")
# video = cv2.VideoCapture(0) # for using CAM

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()
    
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
        
    # Grayscale
    gray = cv2.inRange(frame, (120, 50, 0), (255, 200, 120))
    # chunk into full width rows of pixels
    n_rows = 200
    chunks = [gray[i:i+n_rows,:] for i in range(0, len(gray), n_rows)]

    line_points = []
    

    # for each row, find the middle of the row
    for i in range(len(chunks)):
        chunk = chunks[i]
        points = np.argwhere(chunk == 255)
        
        # if there are no points, skip this row
        if len(points) == 0:
            continue

        # find the middle of the row
        x_avg = np.mean(points[:,1])
        y_avg = np.mean(points[:,0])
        
        # get average deviation
        # low deviation means only one edge was detected
        # high deviation means two edges were detected
        deviation = np.std(points[:,1])
        # if deviation is too low, skip this row
        if deviation < 300:
            continue

        line_points.append((x_avg, y_avg + i*n_rows))
    
    # draw the points
    for point in line_points:
        frame = cv2.circle(frame, (int(point[0]), int(point[1])), 5, (225, 225, 255), -1)

    # fit curve
    if len(line_points) > 1:
        # fit a curve to the points
        x = [point[0] for point in line_points]
        y = [point[1] for point in line_points]
        # maximum gradient is 1
        curve = np.polyfit(x, y, 2)
        # print(curve)
        # calculate the points on the curve
        x_curve = np.linspace(0, frame.shape[1], frame.shape[1])
        y_curve = np.polyval(curve, x_curve)
        # draw the curve
        frame = cv2.polylines(frame, [np.array([x_curve, y_curve]).T.astype(int)], False, (255, 255, 255))


    # show the frame
    cv2.imshow("Tracking", frame)
    # cv2.imshow("Tracking", gray)

    

    
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()