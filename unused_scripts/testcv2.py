import cv2
print(cv2.__version__)

# Create a blank image
image = cv2.imread(r"D:\120140037 - Workspace TA\Inference Repository\YOLOv10n-Implementation-on-Driver-Drowsiness-Detection\VB BPP 3.jpeg")

# Display the image
cv2.imshow('Test Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()