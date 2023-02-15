import sys
import pickle as pkl
import numpy as np
import pandas as pd
import cv2
import os 


def one_hot_encoding(self, labels): 
    num_classes = 10 
    one_hot_labels = np.zeros((len(labels), num_classes)) 
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1

    return one_hot_labels


path_to_folder = sys.argv[1]
data_folder_path = path_to_folder+"/test/"
""" csv_file = path_to_folder + "/test.csv" """

images = []
""" labels = []    

df = pd.read_csv(csv_file)
labels = df['digit'].values """

print("Loading data from ", data_folder_path)
file_names = []

for file in os.listdir(data_folder_path):
    img = cv2.imread(data_folder_path + file) 

    if img is not None:
        file_names.append(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, (80,80))
        l = []
        l.append(np.array(img, dtype=np.float32)/255)
        images.append(l)

print("Dataset dimension ", np.shape(images),"==>",np.shape(images[0]))

images = np.array(images) 
""" labels = np.array(one_hot_encoding(labels)) """
 

model = pkl.load(open("1705102_model.pickle", "rb"))
model.test(images, file_names)