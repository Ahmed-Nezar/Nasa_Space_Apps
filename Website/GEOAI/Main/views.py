from django.shortcuts import render
import os
import torch
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from torchvision import transforms
from PIL import Image


def index_view(request):
    return render(request, 'Main/index.html')


# Define the path to your saved model
MODEL_PATH = r'D:\Nasa_Space_Apps\Website\GEOAI\Prithvi_100M.pt'

# Load your trained model
model = torch.load(MODEL_PATH, map_location= "cpu")  # Replace with your model loading logic


# Define a transformation to apply to the uploaded image
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def your_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']

            # Save the uploaded image to a temporary location
            temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.png')
            with open(temp_image_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)

            # Now, pass the temp_image_path to your model for prediction
            prediction = model(temp_image_path)
            # Clean up the temporary image file
            os.remove(temp_image_path)
            print(prediction)
            return render(request, 'Main/index.html', {'prediction': prediction, 'form': form})

    else:
        form = ImageUploadForm()
    
    return render(request, 'Main/index.html', {'form': form})

# Modify this function to use your loaded model for prediction
def predict_burn_scar(image_path):
    # Load and preprocess the image for prediction
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Use your loaded model for prediction
    with torch.no_grad():
        output = model(image)
    
    # Process the output to obtain a prediction result
    # This is a placeholder, replace with your actual processing based on your model
    prediction = "Prediction: {}".format(output.argmax(dim=1).item())

    return prediction
