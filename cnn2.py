import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN model (same as the one used during training)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model
model_path = "sign_language_cnn.pth"
num_classes = 36  # Update this based on your dataset
class_names = [chr(i) for i in range(97, 123)] + [str(i) for i in range(1, 10)]  # a-z and 1-9
model = CNN(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to preprocess the frame and predict the class
def predict_frame(frame, model, transform, class_names):
    image = Image.fromarray(frame)  # Convert OpenCV frame (numpy array) to PIL image
    image = transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

# OpenCV for real-time video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Flip the frame horizontally (optional, for mirror effect)
    frame = cv2.flip(frame, 1)

    # Define a region of interest (ROI) for hand detection
    x, y, w, h = 300, 300, 600, 600
    roi = frame[y:y+h, x:x+w]

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Predict the class of the ROI
    prediction = predict_frame(roi, model, transform, class_names)

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {prediction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
