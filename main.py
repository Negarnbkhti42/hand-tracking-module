import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from HandDetector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    handLandmarker = HandDetector()

    while True:
        success, img = cap.read()

        handLandmarker.detect_hands(img)
        annoted_image = handLandmarker.draw_landmarks_on_image(
            img, handLandmarker.result
        )

        cv2.imshow("Image", annoted_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    handLandmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
