import base64
import json
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from Levenshtein import distance

load_dotenv(dotenv_path="../.env")


# Constants
IMG_PATH = '../photos'
CORRECT_DATA_JSON='../data/correct.json'
IMAGE_CORR_DATA_JSON='../data/imageCorr.json'
EVALUATION_CSV='../evaluation.csv'
DISTANCE_THRESHOLD = 3

# Initialize API client
api_key = os.getenv("OPEN_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

 

def getOpenAiAssessment(img):
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
                "url": f"data:image/png;base64,{img}"
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

def calculate_combined_accuracy(ground_truth, predictions, threshold=None):
    # Flatten both lists
    flat_ground_truth = [item for sublist in ground_truth for item in sublist]
    flat_predictions = [item for sublist in predictions for item in sublist]

    correct_predictions = []
    incorrect_predictions = []

    # Calculate matches (True Positives) and mismatches
    for gt_item, pred_item in zip(flat_ground_truth, flat_predictions):
        # Apply fuzzy matching if a threshold is provided
        if threshold is not None:
            if distance(gt_item.lower(), pred_item.lower()) <= threshold:
                correct_predictions.append(pred_item)
            else:
                incorrect_predictions.append(pred_item)
        # Apply strict matching if no threshold is provided
        else:
            if gt_item == pred_item:
                correct_predictions.append(pred_item)
            else:
                incorrect_predictions.append(pred_item)

    # Total items considered
    total_items = len(flat_ground_truth)

    # Calculate accuracy
    accuracy = len(correct_predictions) / total_items
    return accuracy, set(correct_predictions), set(incorrect_predictions)

def calculate_set_accuracy(set1, set2):
    # Calculate the intersection and union of the sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the accuracy
    accuracy = len(intersection) / len(union) if union else 0
    return accuracy


def analyzeImages(path):
    imageArr = encode_images_in_directory(path)
    correct_dict, image_corr_dict = getData()

    for image_path, imgData in imageArr:
        correct_key = image_corr_dict[image_path]["correct"]

        # Get the correct matrix
        correct_matrix = correct_dict[correct_key]

        # Get the matrix from the API response
        brandMatrix = extract_matrix(getOpenAiAssessment(imgData))

        bounds = image_corr_dict[image_path]["bounds"]
        width = bounds["x2"] - bounds["x1"]
        height = bounds["y2"] - bounds["y1"]
        correctMatrixAdj = correct_matrix[bounds["y1"]:bounds["y2"]][bounds["x1"]:bounds["x2"]]

        fuzzyScore, fuzzyCorrect, fuzzyIncorrect = calculate_combined_accuracy(correctMatrixAdj, brandMatrix, DISTANCE_THRESHOLD)
        score, correct, incorrect = calculate_combined_accuracy(correctMatrixAdj, brandMatrix)

        correct_brands = {element for row in correctMatrixAdj for element in row}
        observed_brands = {element for row in brandMatrix for element in row}

        evaluationData = {
            "img": image_path,
            "model_type": "GPT-4",
            "accuracy": score,
            "fuzzy_accuracy": fuzzyScore,
            "position_agnostic_accuracy": calculate_set_accuracy(correct_brands, observed_brands),
            "vending_machine_type": correct_key,
            "onAngle": "TODO",
            "incorrect brands": incorrect,
            "correct brands": correct,
            "fuzzy_correct": fuzzyCorrect,
            "fuzzy_incorrect": fuzzyIncorrect,
            "brand matrix": brandMatrix
        }

        addEvaluation(evaluationData)


analyzeImages(IMG_PATH)