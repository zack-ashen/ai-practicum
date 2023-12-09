import base64
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# Initialize API client
api_key = os.getenv("OPEN_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "../photos/drinks2.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "This is a picture of a cooler with drinks in it. It has multiple rows. Send back a matrix array of the brands of the drinks in the cooler. If you are not confident in a particular brand you should just qualify it as 'Unknown'. Do not output the specific type of drink, but rather the brand that makes the drink."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{base64_image}"
            }
          }
        ]
      },
      {
        "role": "system",
        "content": "You're looking for a matrix (array of arrays) format for the brands of drinks. Do not output any text besides the matrix itself."
      },
    ],
    "max_tokens": 300,
    "temperature": 0.0
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

responseText = response.json()['choices'][0]['message']['content']

print(responseText)


def extract_matrix(text):
  # Find the index of the first open bracket '[' and the last closed bracket ']'
    start_index = text.find('[')
    end_index = text.rfind(']')

    # Check if both brackets were found
    if start_index != -1 and end_index != -1 and end_index > start_index:
        # Extract the substring from the first '[' to the last ']'
        content = text[start_index:end_index+1]

        # Convert the string representation of the matrix to a Python list
        try:
            matrix = json.loads(content)
            return matrix
        except json.JSONDecodeError:
            print("Error: The content between the first '[' and the last ']' is not a valid matrix.")
            return None
    else:
        print("Brackets '[' or ']' not found or in incorrect order.")
        return None

print(extract_matrix(responseText))