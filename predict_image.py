import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN model (same architecture as the training one)
class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # conv2
        self.pool = nn.MaxPool2d(2, 2)

        # After two convolution layers, the size of the feature map will be (32, 8, 8)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # fc1 (input size updated)
        self.fc2 = nn.Linear(128, num_classes) # fc2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor to fit into fc1
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),                 # Resize to 32x32
    transforms.Grayscale(num_output_channels=3), # Ensure 3 channels (RGB)
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5),         # Normalize
                         (0.5, 0.5, 0.5))
])

# Define class labels
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load("alzheimer_cnn_model.pth", map_location=device))
model.eval()

# Function to predict a single image
def predict_image(image_path):
    img = Image.open(image_path)

    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Apply transformations
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    return class_names[predicted_class]

# Main function to handle command line inputs and prediction
if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict Alzheimer's Disease Class from MRI Image")
    parser.add_argument('image_path', type=str, help="Alzheimer_MRI_4_classes_dataset/MildDemented/1 (5).jpg")
    args = parser.parse_args()

    # Predict and print class name
    predicted_class = predict_image(args.image_path)
    print(f"Predicted Class: {predicted_class}")

