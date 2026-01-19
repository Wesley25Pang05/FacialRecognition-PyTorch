from PIL import Image
from facenet_pytorch import MTCNN
from embeds import get_embedding
from torchvision import transforms
import numpy

mtcnn = MTCNN(keep_all=True)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def detect_faces(img_input):
    # This allows a file path and/or PIL images
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input

    boxes, _ = mtcnn.detect(img) # Detects the faces

    return boxes

def label_face(face, clf):
    # This labels the face with the correct person,
    # Unknown if the person does not exist in data
    
    face_tensor = transform(face)

    emb = get_embedding(face_tensor)

    probs = clf.predict_proba(emb)[0]
    best_idx = numpy.argmax(probs)
    confidence = probs[best_idx]

    if confidence > 0.85:
        return clf.classes_[best_idx], confidence
    else:
        return "Unknown", confidence