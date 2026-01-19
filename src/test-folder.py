import os
import joblib
from PIL import Image
from detect import detect_faces, label_face

classifier = joblib.load("face_svm.pkl")
edited_count = 0

for filename in os.listdir("data/test"):
    file_path = os.path.join("data/test", filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        img = Image.open(file_path).convert("RGB")
    except Exception:
        continue

    boxes = detect_faces(img)

    if boxes is None or len(boxes) == 0:
        continue

    x1, y1, x2, y2 = map(int, boxes[0])
    face = img.crop((x1, y1, x2, y2))
    label, confidence = label_face(face, classifier)

    edited_count += 1

    new_name = f"{label}_{confidence:.2f}_{edited_count}.jpg"
    new_path = os.path.join("data/test", new_name)

    if not os.path.exists(new_path):
        os.rename(file_path, new_path)

print(f"\nTotal images edited: {edited_count}")
