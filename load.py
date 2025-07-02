import torch
from model import LcdModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
import train
import config

# Initialize the model
model = LcdModel(num_chars=11)  # Replace <NUM_CLASSES> with the correct number

# Load the model weights
model.load_state_dict(torch.load("models/lcd_model_best.pth"))

# Set the model to evaluation mode
model.eval()



# Define the same preprocessing pipeline
transform = A.Compose([
    A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Use the same normalization
    ToTensorV2(),
])

# Read the new image
image_path = "predict\\KWH (6).jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
image = cv2.bitwise_not(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = Image.fromarray(image)
image = np.array(image)

# Apply transformations
augmented = transform(image=image)
input_tensor = augmented["image"].unsqueeze(0)  # Add batch dimension


# Ensure the model and tensor are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():  # No gradient computation during inference
    predictions = model(input_tensor)

targets_flat = ['.','0','1','2','3','4','5','6','7','8','9']
lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flat)
lcd_preds = train.decode_predictions(predictions, encoder=lbl_enc)
print("Predicted Text:", lcd_preds[0])  # Since it's a single image, take the first result
