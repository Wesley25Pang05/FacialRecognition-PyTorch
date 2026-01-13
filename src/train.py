import os
import numpy as np
from PIL import Image
from torchvision import transforms

from classifier import train_svm
from detect import detect_faces
from embeds import get_embedding

embeddings = []
labels = []

# Image preprocessing pipeline for Face Net
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

for label in os.listdir("data/train"): # This goes through each person's folder
    person_dir = os.path.join("data/train", label)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir): # This goes through each image in the current person
        img_path = os.path.join(person_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: # In case if the image is corrupted
            continue

        # Gets the detected faces in the image
        boxes = detect_faces(img_path)

        if boxes is None: # If there is no face it will skip this image in training
            continue

        for box in boxes: # Crops and adds the image to the database for training
            x1, y1, x2, y2 = map(int, box)
            face = img.crop((x1, y1, x2, y2))
            face_tensor = transform(face)

            emb = get_embedding(face_tensor)
            embeddings.append(emb[0])
            labels.append(label)

# Converts embeddings to a numpy array for sklearn
embeddings = np.array(embeddings)

# Train and save the SVM classifier for future use of this project
clf = train_svm(embeddings, labels)

if len(embeddings) > 0:
    clf = train_svm(embeddings, labels)
    print(f"Successful model trained on {len(clf.classes_)} classes")
else:
    print("Training failed: No valid embeddings generated")