import cv2
import time
import numpy as np

def camera():
    # Initialize the camera (0 is usually the default camera)
    camera = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not camera.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        
        if not ret:
            print("Failed to capture frame")
            break
        
        # Display the live feed
        cv2.imshow("Press SPACE to capture, ESC to exit", frame)

        # Wait for key press
        key = cv2.waitKey(1)

        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # SPACE key to capture
            cv2.imwrite("captured_image.jpg", frame)

            # Get frame dimensions
            height, width, _ = frame.shape

            # Create a white frame
            white_frame = np.full((height, width, 3), 255, dtype=np.uint8)
 
            # Show white screen for a brief moment
            cv2.imshow("Press SPACE to capture, ESC to exit", white_frame)
            cv2.waitKey(100)  # Show the white frame for 100ms

            # Fade-out effect, I don't fully understand this but shout out chatGPT
            for alpha in np.linspace(1, 0, num=10):  # Smooth fade-out
                fade_frame = cv2.addWeighted(white_frame, alpha, frame, 1 - alpha, 0)
                cv2.imshow("Press SPACE to capture, ESC to exit", fade_frame)
                cv2.waitKey(50)  # Adjust speed of fade-out

            time.sleep(1)
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()


camera()