import torch
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face_tensor):
    face_tensor = face_tensor.to(device) # Moves the tesnor to the device model
    with torch.no_grad():
        return model(face_tensor.unsqueeze(0)).cpu().numpy() # Removes gradients and returns as an array