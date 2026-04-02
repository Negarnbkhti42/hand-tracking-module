import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from HandDetector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    hand_landmarker = HandDetector(min_num_hands=2)

    while True:
        _, img = cap.read()

        hand_landmarker.detect_hands(img)
        annotated_image = hand_landmarker.draw_landmarks_on_image(
            img, hand_landmarker.result
        )

        cv2.imshow("Image", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
