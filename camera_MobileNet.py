import torch
import torch.nn as nn
from torchvision import models
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

ATTR = "Smile"
MODEL = "smile_final.pth"

# Odtworzenie struktury modelu
class SmileClassifier(nn.Module):
    def __init__(self, base_model):
        super(SmileClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1000, 128),  # Zgodnie z kodem treningowym
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.base_model(x)
        return self.classifier(features)
    
def enlarge_bounding_box(x, y, w, h, scale=1.2, frame_width=None, frame_height=None, shift_up=0):
    # Calculate new width and height
    w_new = int(w * scale)
    h_new = int(h * scale)

    # Adjust top-left corner to keep the bounding box centered
    x_new = x - (w_new - w) // 2
    y_new = y - (h_new - h) // 2 - shift_up  # Shift upward by shift_up pixels

    # Ensure the bounding box is within the frame
    if frame_width is not None:
        x_new = max(0, x_new)
        w_new = min(frame_width - x_new, w_new)
    if frame_height is not None:
        y_new = max(0, y_new)
        h_new = min(frame_height - y_new, h_new)

    return x_new, y_new, w_new, h_new


# Transformacje obrazu
transform_live = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Funkcja do klasyfikacji
def classify_face(face, model, device, attr):
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    input_tensor = transform_live(face_pil).unsqueeze(0).to(device)

    # Convert the transformed tensor back to a numpy array for visualization
    transformed_image = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # H x W x C
    transformed_image = ((transformed_image * 0.229) + 0.485) * 255  # De-normalize and scale to 0-255
    transformed_image = transformed_image.astype(np.uint8)

    # Display the transformed image
    cv2.imshow("Transformed 224x224 Face", cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Apply a threshold to classify
    if prediction >= 0.6:  # Confidence threshold
        return attr  # Positive class (e.g., "Male")
    elif prediction <= 0.4:  # Confidence threshold for negative class
        return f"No {attr}"  # Negative class (e.g., "No Male")
    else:
        return "Unknown"  # Low confidence

# Wykrywanie twarzy
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))
    return faces

# Rysowanie twarzy i etykiet
def draw(frame, faces, predictions):
    for (x, y, w, h), pred in zip(faces, predictions):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Główna pętla
if __name__ == "__main__":
    # Wczytanie modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    base_model = models.mobilenet_v2(pretrained=True)  # Bez modyfikacji
    model = SmileClassifier(base_model).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))

    face_cascade = cv2.CascadeClassifier("face.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    skip = 5
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % skip == 0:
            faces = detect_faces(frame, face_cascade)
            enlarged_faces = []
            attributes = []

            for (x, y, w, h) in faces:
                x, y, w, h = enlarge_bounding_box(
                    x, y, w, h, scale=2.5, frame_width=frame.shape[1], frame_height=frame.shape[0], shift_up=50)
                enlarged_faces.append((x, y, w, h))

                face = frame[max(0, y):min(y + h, frame.shape[0]), max(0, x):min(x + w, frame.shape[1])]
                if face.size == 0:
                    attributes.append("Unknown")
                    continue

                gender = classify_face(face, model, device, ATTR)
                attributes.append(gender)

        draw(frame, enlarged_faces, attributes)
        cv2.imshow("Smile Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
