import cv2

img_array = np.array(img)

# Resize image
img_resized = cv2.resize(img_array, (640, 640))

results = model.predict(img_resized, conf=0.25)
