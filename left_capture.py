import cv2
import numpy

IMAGE_RGB_PATH = "data_baru/data{:01d}.jpg"

# Open the ZED camera

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if cap.isOpened() == 0:
    exit(-1)

frameId = 0
while True :
    # Get a new frame from camera
    retval, frame = cap.read()
    # Extract left and right images from side-by-side
    left_right_image = numpy.split(frame, 2, axis=1)
    # Display images
    #cv2.imshow("frame", frame)
    cv2.imshow("left", left_right_image[0])
    #cv2.imshow("right", left_right_image[1])
    k = cv2.waitKey(1)
    
    if k == ord('x'):
                cv2.destroyAllWindows()
                imageToDisplay = 0
                print 'Exit Program'
                break
    elif k == ord('c'):
                  cv2.imwrite(IMAGE_RGB_PATH.format(frameId), left_right_image[0])
         #        cv2.imwrite(IMAGE_GRAY_PATH.format(frameId), gray)
                  print 'Capture %d'%(frameId)
                  frameId += 1

	# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

exit(0)
