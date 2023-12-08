from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import \
    VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Replace with your Azure subscription key and endpoint


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

