import cv2
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms

from detect import detect_faces
from embeds import get_embedding

clf = joblib.load("face_svm.pkl")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

THRESHOLD = 0.85

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes = detect_faces(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = pil_img.crop((x1, y1, x2, y2))
            face_tensor = transform(face)

            emb = get_embedding(face_tensor)

            probs = clf.predict_proba(emb)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            label = clf.classes_[best_idx]

            if confidence < THRESHOLD:
                label = "Unknown"

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

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()