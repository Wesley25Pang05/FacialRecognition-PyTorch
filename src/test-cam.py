import cv2
import joblib
from PIL import Image
from detect import detect_faces, label_face

classifier = joblib.load("face_svm.pkl")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes = detect_faces(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = pil_img.crop((x1, y1, x2, y2))
            label, confidence = label_face(face, classifier)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Recognition Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

camera.release()
cv2.destroyAllWindows()