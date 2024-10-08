# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:25:22 2023

@author: Katarina
"""

import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
import collections
import math
import csv
from scipy import ndimage
from sklearn.cluster import KMeans
import os
# keras
from tensorflow import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        
        
def dilate(image):
    kernel_height = 8
    kernel_width = 1
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_to_range(image):
    return image/255

# flatiranje
def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet, output_size):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(output_size)
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1, x:x+w+1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) 
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def hamming_distance(word1, word2):
    if len(word1) != len(word2):
        min_len = min(len(word1), len(word2))
        return sum(c1 != c2 for c1, c2 in zip(word1[:min_len], word2[:min_len])) + abs(len(word1) - len(word2))
    else:
        return sum(c1 != c2 for c1, c2 in zip(word1, word2))


def display_result_with_spaces(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def apply_k_means(distances):
    if len(distances) >= 2:
        k_means = KMeans(n_clusters=2, n_init=10)
        k_means.fit(distances)
        return k_means
    else:
        print("Not enough samples for clustering.")
        return None

def get_unique_letters(letters):
    unique_letters = []
    unique_letters_indices = []

    for idx, region in enumerate(letters):
        if idx == 0:
            unique_letters.append(region)
            unique_letters_indices.append(idx)
        else:
            # Check if the current region is sufficiently distant from all previous regions
            is_unique = all(hamming_distance(region.flatten(), prev_region.flatten()) > 100 for prev_region in unique_letters)
            if is_unique:
                unique_letters.append(region)
                unique_letters_indices.append(idx)

    # Prikazivanje slika unikatnih slova
    for idx in unique_letters_indices:
        plt.figure()
        display_image(letters[idx], color=False)
        plt.title(f"Unique Letter {idx}")
        plt.show()

    return unique_letters


def load_correct_words(file_path):
    correct_words = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                image_filename, word = row
                correct_words[image_filename] = word
    return correct_words


alphabet = ['г', 'о', 'л', 'у', 'б', 'й', 'э', 'к', 'р', 'а','н']

def print_csv_content(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        header = next(reader)  
        for row in reader:
            print(row)

# Replace with the actual file path
csv_file_path = r'C:\Users\Katarina\Desktop\softk3\data1\res.csv'

# Call the function to print the content of the CSV file
print_csv_content(csv_file_path, encoding='utf-8')  # or encoding='cp1251'


def process_image(image_path):
    # Load and process the image
    image_color = load_image(image_path)
    image_color = image_color[:255, :]
    img = image_bin(image_gray(image_color))
    img = invert(img)
    img = erode(dilate(img))
    selected_regions, letters, _ = select_roi_with_distances(image_color.copy(), img)

    unique_letters = get_unique_letters(letters)

    return image_color, unique_letters

def display_images_in_directory(directory_path):
    # List all files in the directory
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Initialize lists to store training data
    all_training_inputs = []
    all_training_outputs = []

    # Iterate through each image file and process it
    for image_file in image_files:
        # Create the full path to the image file
        image_path = os.path.join(directory_path, image_file)

        # Process the image and get training data
        _, unique_letters = process_image(image_path)

        # Pripremi trening skup
        training_inputs = prepare_for_ann(unique_letters)
        training_outputs = convert_output(alphabet, output_size=len(alphabet))

        # Append to the overall training data
        all_training_inputs.extend(training_inputs)
        all_training_outputs.extend(training_outputs)

    # Train the model using all collected data
    ann = create_ann(output_size=len(alphabet))
    ann = train_ann(ann, all_training_inputs, all_training_outputs, epochs=2000)

    # Now, iterate through each image file again for testing
    for image_file in image_files:
        # Create the full path to the image file
        image_path = os.path.join(directory_path, image_file)

        # Process the image and get testing data
        image_color, _ = process_image(image_path)
        selected_regions_test, letters_test, distances_test = select_roi_with_distances(image_color.copy(), image_color)

        inputs_test = prepare_for_ann(letters_test)
        results_test = ann.predict(np.array(inputs_test, np.float32))

        # Koristi KMeans za dodavanje razmaka
        distances_test = np.array(distances_test).reshape(len(distances_test), 1)
        k_means_test = apply_k_means(distances_test)

        if k_means_test is not None:
            print(display_result_with_spaces(results_test, alphabet, k_means_test))

        true_text = "голубой экран"

        recognized_text = display_result_with_spaces(results_test, alphabet, k_means_test)

        distance = hamming_distance(true_text, recognized_text)

        print(f"Tačan tekst: {true_text}")
        print(f"Prepoznati tekst: {recognized_text}")
        print(f"Hammingovo rastojanje: {distance}")

        plt.figure()
        display_image(image_color, color=True)
        plt.title(f"Image: {image_file}")
        plt.show()

# Replace with the actual directory path
image_directory_path = r'C:\Users\Katarina\Desktop\softk3\data1\pictures'

# Call the function to display images in the specified directory
display_images_in_directory(image_directory_path)