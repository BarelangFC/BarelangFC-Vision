import numpy as np
import cv2

IMAGE_RGB_PATH = "../DOcuments/data_{:01d}.jpg"
#IMAGE_GRAY_PATH = "../getData/Gray/gambar_{:01d}.jpg"

def main():
	cap = cv2.VideoCapture(0)
	frameId = 0

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

	#	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.imshow('frame',frame)
		#cv2.imshow('grayscale',gray)

		# Waiting keyboard interrupt
                k = cv2.waitKey(1)
		if k == ord('x'):
                        cv2.destroyAllWindows()
                        imageToDisplay = 0
                        print 'Exit Program'
                        break
		elif k == ord('c'):
                        cv2.imwrite(IMAGE_RGB_PATH.format(frameId), frame)
         #               cv2.imwrite(IMAGE_GRAY_PATH.format(frameId), gray)
                        print 'Capture %d'%(frameId)
                        frameId += 1

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
        print 'Running BarelangFC-Vision Get Data'
        main()
