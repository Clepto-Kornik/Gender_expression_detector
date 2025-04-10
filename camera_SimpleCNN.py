import cv2
import torch
import numpy as np
from torchvision import transforms
from utils import *
import torch.nn.functional as F
from PIL import Image
from utils import *

MODEL = "test_male.pth"
ATTR = "Male"

# Updated preprocessing pipeline for live input
transform_live = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

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

def my_classify(face, model, device, attr):
    # Convert the face (numpy array) to a PIL image
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL

    # Apply the transformation (resize, normalization)
    input_tensor = transform_live(face_pil).unsqueeze(0).to(device)

    # Convert the transformed tensor back to a numpy array for visualization
    transformed_image = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # H x W x C
    transformed_image = ((transformed_image * 0.229) + 0.485) * 255  # De-normalize and scale to 0-255
    transformed_image = transformed_image.astype(np.uint8)

    # Display the transformed image
    cv2.imshow("Transformed 128x128 Face", cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

    # Perform the gender classification
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    # Calculate probabilities
    probabilities = F.softmax(prediction, dim=1)
    p_no = probabilities[0, 0].item()
    p_yes = probabilities[0, 1].item()

    if max(p_no, p_yes) < 0.6:  # Confidence threshold
        return "Unknown"
    return "No " + attr if p_no > p_yes else attr

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))
    return faces

def draw(frame, faces, genders):
    for (x, y, w, h), gender in zip(faces, genders):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleCNN().to(device)
    state_dict = torch.load(MODEL, map_location=device, weights_only=True)  # Safe loading
    model.load_state_dict(state_dict)

    face_cascade = cv2.CascadeClassifier("face.xml")
    cap = cv2.VideoCapture(0)  # Ensure the correct camera index
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
                    x, y, w, h, scale=2.3, frame_width=frame.shape[1], frame_height=frame.shape[0], shift_up=50)
                enlarged_faces.append((x, y, w, h))

                face = frame[max(0, y):min(y + h, frame.shape[0]), max(0, x):min(x + w, frame.shape[1])]
                if face.size == 0:
                    attributes.append("Unknown")
                    continue

                gender = my_classify(face, model, device, ATTR)
                attributes.append(gender)

        draw(frame, enlarged_faces, attributes)
        cv2.imshow("Face and Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
