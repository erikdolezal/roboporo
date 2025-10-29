from ctu_crs import CRS93 # or CRS93
import cv2

# Initialize the robot interface
robot = CRS93()
robot.initialize()

# Grab an image from the camera
# This will automatically connect to and open the camera on the first call
img = robot.grab_image()

# 'img' is a NumPy array containing the image data.
# You can now process it, for example, using a library like OpenCV.
if img is not None:
    print(f"Successfully grabbed image with shape: {img.shape}")
    cv2.imshow("Robot Camera Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Don't forget to close the connection when you're done
robot.close()