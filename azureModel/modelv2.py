import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import \
    VisualFeatureTypes
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials

load_dotenv(dotenv_path="../.env")

# Replace with your Azure subscription key and endpoint
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

# Create a Computer Vision client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Define the image URL
image_url = "https://pepsimidamerica.com/wp-content/uploads/2021/03/pepsi-mid-america-marion-vending-machines-alt.png"

# Analyze the image for brands
analysis = client.analyze_image(image_url, visual_features=[VisualFeatureTypes.brands])

# Check if any brands were detected and print them
if analysis.brands:
    for brand in analysis.brands:
        print(f"Brand: {brand.name}, Confidence: {brand.confidence}")
else:
    print("No brands detected.")

