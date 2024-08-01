import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

# Load the pre-trained EfficientNetV2 model
effnetv2 = timm.create_model('tf_efficientnetv2_s', pretrained=True, features_only=True)

# Set the model to evaluation mode
effnetv2.eval()

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




# Function to predict image
def predict_image(image_path, model, preprocess):
    # Open the image
    img = Image.open(image_path)
    
    # Apply preprocessing
    img_tensor = preprocess(img)
    
    # Add batch dimension
    # img_tensor = img_tensor.unsqueeze(0)
    img_tensor = torch.rand(1,3,1024,512)
    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert output probabilities to predicted class index
    _, predicted_idx = torch.max(output, 1)
    
    return predicted_idx.item()

# Path to the image of a car
image_path = "cat.jpg"

# Predict the image
predicted_class = predict_image(image_path, effnetv2, preprocess)
print("Predicted class index:", predicted_class)

