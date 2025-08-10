import easyocr
import cv2
import pyttsx3
import numpy as np

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize text-to-speech
engine = pyttsx3.init()

# Open webcam
cap = cv2.VideoCapture(0)

last_texts = set()  # to avoid repeating the same text

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OCR on the frame
    results = reader.readtext(frame)

    current_texts = set()

    for (bbox, text, prob) in results:
        # Convert bbox to rectangle
        points = np.array(bbox).astype(int)
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        current_texts.add(text)

    # Speak only new text
    new_texts = current_texts - last_texts
    for t in new_texts:
        engine.say(t)
    if new_texts:
        engine.runAndWait()

    last_texts = current_texts

    cv2.imshow("OCR Reader", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()