import cv2
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="c:\\data\\frc_images")
parser.add_argument('--prefix', default = str(random.randint(0, 10000)))

args = parser.parse_args()
folder = args.folder
os.makedirs(folder, exist_ok=True)


# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(0)

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print('Frames per second : ', fps,'FPS')

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(7)
    print('Frame count : ', frame_count)

count = 0
while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vid_capture.read()
    # flip image horizontally
    if ret == True:
        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame',frame)

        filename = os.path.join(folder, f'i_{args.prefix}_{count}.png')
        cv2.imwrite(filename, frame)
        count += 1
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(100)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()