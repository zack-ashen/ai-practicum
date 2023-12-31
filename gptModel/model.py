import base64
import json
import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from Levenshtein import distance

load_dotenv(dotenv_path="../.env")


# Constants
IMG_PATH = '../photosv2'
CORRECT_DATA_JSON='../data/correct.json'
IMAGE_CORR_DATA_JSON='../data/imageCorr.json'
EVALUATION_CSV='../evaluation2.csv'
DISTANCE_THRESHOLD = 5

# Initialize API client
api_key = os.getenv("OPEN_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

 

def getOpenAiAssessment(img, parameterized=(None, None)):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }

  x, y = parameterized


  systemPrompt = f"You're looking for a {x}x{y} matrix (array of arrays) format for the brands of drinks this implies there should be {x}*{y} brands or entries in the matrix. Do not output any text besides the matrix itself." if x else "You're looking for a matrix (array of arrays) format for the brands of items. Do not output any text besides the matrix itself. Each array in the matrix should be a row of the vending machine with each element in the array being the brand of the drink as a string."

  payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Please analyze the image of the vending machine and output a matrix. \
                        The matrix should consist of arrays representing each row of products \
                        in the vending machine. Each array should contain strings of the brand names visible in \
                        that row in the position they appear from left to right. If there is nothing in a vending machine slot put 'Unknown' in that position.\
                        In addition, if you don't know a brand name, put 'Unknown' in that position. \
                        Each array should be the same length as the number of columns in the vending machine. If you don't know the number of columns, put 'Unknown' in that position."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/png;base64,{img}"
              }
            }
          ]
        },
        {
          "role": "system",
          "content": systemPrompt
        },
      ],
      "max_tokens": 300,
      "temperature": 0.0
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response.json()['choices'][0]['message']['content']


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
            return None
    else:
        return None
    
def encode_images_in_directory(directory_path):
    encoded_images = []
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        # Check if the file is an image
        if os.path.isfile(image_path) and is_image(filename):
            # Encode the image
            encoded_image = encode_image(image_path)
            encoded_images.append((image_path, encoded_image))
    return encoded_images

def is_image(filename):
    # You can customize this function to check for specific file extensions
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))



encoded_images = encode_images_in_directory(IMG_PATH)


def getData():
  # Open the JSON file
  with open(CORRECT_DATA_JSON, "r") as f:
      # Read the contents of the file
      correct_json = f.read()

  # Open the JSON file
  with open(IMAGE_CORR_DATA_JSON, "r") as f:
      # Read the contents of the file
      image_corr_json = f.read()

  # Convert JSON string to dictionary
  return json.loads(correct_json), json.loads(image_corr_json)


def addEvaluation(evaluationData):
    # Read the CSV file
    data = pd.read_csv(EVALUATION_CSV)

    # Convert evaluationData to DataFrame if it's not already
    if not isinstance(evaluationData, pd.DataFrame):
        evaluationData = pd.DataFrame([evaluationData])

    # Append the new data as a new row
    data = pd.concat([data, evaluationData], ignore_index=True)

    # Save the updated data back to the CSV file
    data.to_csv(EVALUATION_CSV, index=False)

def accuracy(correct_matrix, brand_matrix, fuzzy=False):
    accuracyMap = {}

    totalBrands = 0
    for i in range(len(correct_matrix)):
        for j in range(len(correct_matrix[i])):
            correct_brand = correct_matrix[i][j]
            totalBrands += 1

            if (i, j) not in accuracyMap:
                accuracyMap[(i, j)] = correct_brand
    
    correct = 0
    correct_brands = set()
    incorrect_brands = set()
    for i in range(len(brand_matrix)):
        for j in range(len(brand_matrix[i])):
            brand = brand_matrix[i][j]
            if (i, j) in accuracyMap:
                correct_brand = accuracyMap[(i, j)]
                if (fuzzy and distance(correct_brand, brand) <= DISTANCE_THRESHOLD) or (not fuzzy and correct_brand == brand):
                    correct_brands.add(correct_brand)
                    correct += 1
                else:
                    incorrect_brands.add(correct_brand)

    return (correct / totalBrands), correct_brands, incorrect_brands

def calculate_set_accuracy(set1, set2):
    intersection = len(set1.intersection(set2))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union


def matrix_to_list(matrix):
    return {item for sublist in matrix for item in sublist}


def analyzeImages(path, parameterized=False):
    imageArr = encode_images_in_directory(path)
    correct_dict, image_corr_dict = getData()

    for image_path, imgData in imageArr:
        correct_key = image_corr_dict[image_path]["correct"]

        # Get the correct matrix
        correct_matrix = correct_dict[correct_key]

        # Get the matrix from the API response
        bounds = image_corr_dict[image_path]["bounds"]
        width = bounds["x2"] - bounds["x1"]
        height = bounds["y2"] - bounds["y1"]

        brandMatrix = [[]]
        if (parameterized):
            brandMatrix = extract_matrix(getOpenAiAssessment(imgData, parameterized=(width, height)))
        else:
            brandMatrix = extract_matrix(getOpenAiAssessment(imgData))
        

        correctMatrixAdj = correct_matrix[bounds["y1"]:bounds["y2"]][bounds["x1"]:bounds["x2"]]

        fuzzyScore, fuzzyCorrect, fuzzyIncorrect = accuracy(correctMatrixAdj, brandMatrix, True)
        score, correct, incorrect = accuracy(correctMatrixAdj, brandMatrix)

        correct_brands = matrix_to_list(correctMatrixAdj)
        observed_brands = matrix_to_list(brandMatrix)



        evaluationData = {
            "Path": image_path,
            "Model": "GPT-4",
            "Accuracy": score,
            "Fuzzy Accuracy": fuzzyScore,
            "Jaccard Accuracy": calculate_set_accuracy(correct_brands, observed_brands),
            "Vending Machine Type": correct_key,
            "Correct": correct,
            "Incorrect": incorrect,
            "Fuzzy Correct": fuzzyCorrect,
            "Fuzzy Incorrect": fuzzyIncorrect,
            "Brand Matrix": brandMatrix,
            "Parametrized": parameterized,
        }

        addEvaluation(evaluationData)


analyzeImages(IMG_PATH)
analyzeImages(IMG_PATH, parameterized=True)