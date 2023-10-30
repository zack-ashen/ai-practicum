import openai
import os
from dotenv import load_dotenv
load_dotenv()


# Initialize API client
openai.api_key = os.getenv("OPEN_API_KEY")

# Prepare your image
with open("path_to_your_image.jpg", "rb") as f:
    image = f.read()

# Use the API to classify or describe the image
response = openai.Image.create(
    model="image-model-id",  # Replace with the appropriate model ID for image classification
    data=image
)

# Print the response or description
print(response['data']['text'])