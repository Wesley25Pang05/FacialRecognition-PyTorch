from PIL import Image
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True)

def detect_faces(img_input):
    # This allows a file path and/or PIL images
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input

    boxes, _ = mtcnn.detect(img) # Detects the faces

    return boxes