import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import \
    VisualFeatureTypes
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials

load_dotenv(dotenv_path="../.env")

# Replace with your Azure subscription key and endpoint
subscription_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

# Create a Computer Vision client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Define the path to your local image
local_image_path = '../photos/drinks2.png'

# Open the local image as a binary file
with open(local_image_path, "rb") as image_stream:
    # Analyze the image for brands
    analysis = client.analyze_image_in_stream(image_stream, visual_features=[VisualFeatureTypes.brands])

print(analysis)

# Check if any brands were detected and print them
if analysis.brands:
    for brand in analysis.brands:
        print(f"Brand: {brand.name}, Confidence: {brand.confidence}")
else:
    print("No brands detected.")