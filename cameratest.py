import cv2

# 1. Initialize the camera (0 is usually the default webcam)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Camera Preview")

while True:
    # 2. Capture frame-by-frame
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 3. Display the resulting frame
    cv2.imshow("Camera Preview", frame)

    # 4. Wait for key press
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        # ESC pressed to quit
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed to save image
        img_name = "captured_image.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}!")
        break

# 5. Release resources
cam.release()
cv2.destroyAllWindows()
