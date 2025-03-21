import cv2  
import numpy as np  
import winsound  

def motion_detector():
    cap = cv2.VideoCapture(0)

    # Define the motion detection area
    x1, y1, x2, y2 = 100, 100, 400, 400  

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply background subtraction
        fgmask = fgbg.apply(gray)

        # Threshold the image
        _, thresh = cv2.threshold(fgmask, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the detection area
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small movements
                x, y, w, h = cv2.boundingRect(contour)

                # Check if motion is inside the defined rectangle
                if x1 < x + w // 2 < x2 and y1 < y + h // 2 < y2:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Motion Detected!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    winsound.Beep(1000, 200)  # Alert sound

        # Display output
        cv2.imshow("Motion Detector", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    motion_detector()
