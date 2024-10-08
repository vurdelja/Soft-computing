# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:49:39 2023

@author: Katarina
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt  
import sys
import os
from sklearn.metrics import mean_absolute_error


def countSquirtles(pictureFolder):
    counted_pokemons = {}
    for filename in os.listdir(pictureFolder):
        f = os.path.join(pictureFolder, filename)
        if not os.path.isfile(f):
            break


        img = cv2.imread(f)  
        plt.imshow(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        plt.imshow(img)

               
        img = img[60:, :]  
        plt.imshow(img) 
               
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray.shape
        plt.imshow(img_gray, 'gray')  
               
        image_ada_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 87, 7)
        plt.imshow(image_ada_bin, 'gray')



        kernel = np.ones((3, 3)) 

        image_ada_bin = cv2.erode(image_ada_bin, kernel, iterations=2)
        image_ada_bin = cv2.dilate(image_ada_bin, kernel, iterations=1)

        plt.imshow(image_ada_bin, 'gray')
        plt.show()
               
               
        contours, hierarchy = cv2.findContours(image_ada_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_c = img.copy()

        squirtles = []  # ovde ce biti samo konture koje pripadaju pokemonu
        for contour in contours: 
                    center, size, angle = cv2.minAreaRect(contour)
                    width, height = size
                    if width > 19 and width < 120 and height > 16 and height < 90:
                        squirtles.append(contour)
        cv2.drawContours(img_c, squirtles, -1, (255, 0, 0), 1)
        plt.imshow(img_c)
        plt.show()
        counted_pokemons[filename] = (len(squirtles))
    return counted_pokemons


def ispis(counted_pokemons):
    correct = {}
    with open("squirtle_count.csv") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            lineParts = line.split(',')
            value = lineParts[1].rsplit()
            correct[lineParts[0]] = int(value[0])

    for file_name in correct:
        print(file_name + '-' + str(correct[file_name]) + '-' + str(counted_pokemons[file_name]))

    correct_list = list(correct.values())
    counted_pokemons_list = list(counted_pokemons.values())
    mae = mean_absolute_error(correct_list, counted_pokemons_list)
    print(mae)
    return mae


if len(sys.argv) > 1:
    pictureFolder = sys.argv[1]
else:
    pictureFolder = 'pictures1'

counted_pokemons = countSquirtles(pictureFolder)
ispis(counted_pokemons)