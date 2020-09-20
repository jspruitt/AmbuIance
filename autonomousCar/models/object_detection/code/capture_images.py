# import cv2 as cv
# import logging
# 
# cap = cv.VideoCapture(-1)
# 
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# 
# i = 0    
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#         
#     # Our operations on the frame come here
#     cv.imshow('frame', frame)
#     
#     if cv.waitKey(1) == ord('q'):
#         break
#     
#     # Pressing w takes a picture
#     if cv.waitKey(1) == ord('w'):
#         cv.imwrite(str(i) + ".png", frame)
#         i += 1
#         print(str(i) + ". Took a picture")
#         cv.waitKey(5000)
#         
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()