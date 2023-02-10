import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import scipy
import sklearn
import tqdm
import cv2
import os



class Convolution :
    def __init__(self, N_out_channel, filter_dimension, stride, padding):
        self.N_out_channel = N_out_channel
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.padding = padding
        self.kernels = []
        self.X = []
        self.shape = None
        for i in range(N_out_channel):
            self.kernels.append(np.random.randint(1,10, size=(filter_dimension, filter_dimension)))
        self.kernels = np.array(self.kernels)
        self.biases = np.zeros(N_out_channel) 
        self.scale_constant = 0.0001
        self.alpha = 0.0001
        

    def convolve(self, image, kernel,dim_x, dim_y):
        output = np.zeros((dim_x, dim_y))
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                conv_box = image[x*self.stride:x*self.stride+kernel.shape[0], y*self.stride:y*self.stride+kernel.shape[1]]
                if conv_box.shape != kernel.shape: 
                    continue
                output[x,y] = np.sum(conv_box*kernel)
        return output

    def add_padding(self, image, padding_x, padding_y):
        if padding_x <= 0 & padding_y <= 0:
            return image
        return np.pad(image, [(padding_x, padding_y),(padding_x, padding_y)], mode='constant') 

    def forward_propagation(self, images):  
        print("Convolution ",end=" ")

        self.X = images
        feature_map_dimX = (int)((images[0][0].shape[0]-self.filter_dimension+self.padding+self.stride)/self.stride)
        feature_map_dimY = (int)((images[0][0].shape[1]-self.filter_dimension+self.padding+self.stride)/self.stride)
        feature_map = np.zeros((np.shape(images)[0], self.N_out_channel, feature_map_dimX, feature_map_dimY))
        self.shape = feature_map.shape
        
        for image_idx in range(np.shape(images)[0]):
            for k in range(np.shape(images)[1]):
                image = self.add_padding(images[image_idx][k] , self.padding, self.padding)
                self.X[image_idx][k] = image
                for i in range(self.N_out_channel): 
                    feature_map[image_idx][i] = self.convolve(image, self.kernels[i], feature_map_dimX, feature_map_dimY) + self.biases[i]
        return feature_map

    def back_propagation(self, del_Z): 
        print("Convolution ",end=" ")
        del_K = np.zeros(self.kernels.shape) 

        for idx in range(del_K.shape[0]):
            for k in range(np.shape(self.X)[0]): 
                for i in range(np.shape(self.X)[1]):
                    image = self.X[k][i]
                    del_K[idx] += self.convolve(image, del_Z[k][idx], del_K.shape[1], del_K.shape[2]) 

        del_B = np.sum(np.sum(np.sum(del_Z, axis=3),axis = 2), axis=0)

        rotated_kernels = np.array([np.fliplr(np.flipud(kernel)) for kernel in self.kernels]) 
        del_X = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], np.shape(self.X[0][0])[0], np.shape(self.X[0][0])[1]))
        print("\n del_X shape ", np.shape(del_X)," del_Z shape ", np.shape(del_Z))

        padding_x =((np.shape(del_Z)[2]*self.stride) - np.shape(del_X)[2]) + self.filter_dimension - self.stride - self.padding
        padding_y =((np.shape(del_Z)[3]*self.stride) - np.shape(del_X)[3]) + self.filter_dimension - self.stride - self.padding
        
        print("padding_x ", padding_x, " padding_y ", padding_y)

        for k in range(np.shape(del_Z)[0]): 
            for i in range(np.shape(del_Z)[1]):
                del_Z[k][i] = self.add_padding(del_Z[k][i], padding_x, padding_y)

        
        for k in range(np.shape(del_X)[0]): 
            for j in range(np.shape(del_X)[1]):
                for i in range(np.shape(del_Z)[1]): 
                    for idx in range(rotated_kernels.shape[0]):
                        del_X[k][j] += self.convolve(del_Z[k][i], rotated_kernels[idx], np.shape(del_X[k][j])[0], np.shape(del_X[k][j])[1])

        self.kernels = self.kernels - self.alpha*del_K
        self.biases = self.biases - self.alpha*del_B
        return del_X

        

class ReLU:
    def __init__(self):
        self.RelU_input = []
        self.shape = None
    def forward_propagation(self, feature_map): 
        print("ReLU ",end=" ")
        self.ReLU_input = feature_map
        self.shape = feature_map.shape
        ReLU_output = np.zeros(feature_map.shape)
        for img_idx in range(feature_map.shape[0]):
            for i in range(feature_map.shape[1]):
                for x in range(feature_map.shape[2]):
                    for y in range(feature_map.shape[3]):
                        ReLU_output[img_idx][i][x][y] = max(0, feature_map[img_idx][i][x][y]) 
        return ReLU_output

    def back_propagation(self, del_C): 
        print("ReLU ",end=" ")
        del_C_by_del_Z = np.zeros(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for x in range(self.shape[2]):
                    for y in range(self.shape[3]):
                        if self.ReLU_input[i,j,x,y] > 0:
                            del_C_by_del_Z[i,j,x,y] = 1
        return del_C_by_del_Z * del_C 
        


class Pooling:
    def __init__(self, pool_dimension, stride):
        self.pool_dimension = pool_dimension
        self.stride = stride
        self.pool_input = []
        self.shape = None
        self.del_p = []
    
    def forward_propagation(self, feature_map): 
        print("Pooling ",end=" ")
        pool_dimX =(int) ((feature_map[0][0].shape[0]-self.pool_dimension+self.stride)/self.stride)
        pool_dimY =(int) ((feature_map[0][0].shape[1]-self.pool_dimension+self.stride)/self.stride)
        pool = np.zeros((feature_map.shape[0],feature_map.shape[1], pool_dimX, pool_dimY))
        self.shape = pool.shape
        self.input_shape = feature_map.shape
        for img_idx in range(feature_map.shape[0]):
            for i in range(feature_map.shape[1]):
                for x in range(pool_dimX):
                    for y in range(pool_dimY):
                        pool[img_idx][i][x][y] = np.max(feature_map[img_idx][i][(x*self.stride) : (x*self.stride+self.pool_dimension), (y*self.stride) : (y*self.stride+self.pool_dimension)])
        self.pool_input = feature_map
        return pool

    def back_propagation(self, del_Z): 
        print("Pooling ",end=" ")
        del_P = np.array(del_Z)
        del_C = np.zeros(shape=self.input_shape) 
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for x in range(del_P.shape[2]) :
                    for y in range(del_P.shape[3]):
                        window = np.array(self.pool_input[i][j][(x*self.stride) : (x*self.stride+self.pool_dimension), (y*self.stride) : (y*self.stride+self.pool_dimension)])
                        max_index = np.unravel_index(window.argmax(), window.shape) 
                        del_C[i][j][(x*self.stride) + max_index[0]][(y*self.stride) + max_index[1]] = del_P[i][j][x][y]


        return del_C
        

class FlattenLayer:
    def __init__(self):
        self.flattened_vector = []
        self.del_f = []

    
    def normalize(self, x):
        x_hat = (x - np.mean(x)) / np.std(x)
        return x_hat

    def forward_propagation(self, pool): 
        print("Flatten ",end=" ")
        self.pool_shape = pool.shape
        pool = np.array(pool)
        flattened_vector = []
        for img_idx in range(pool.shape[0]):
            flattened_vector.append(self.normalize(pool[img_idx].flatten()))
        self.flattened_vector = np.array(flattened_vector)
        return self.flattened_vector

    def back_propagation(self, del_Z): 
        print("Flatten ",end=" ")
        self.del_f =  np.reshape(del_Z, self.pool_shape)
        return self.del_f

class FullyConnectedNN:
    def __init__(self, output_dim):
        self.output_dim = output_dim 
        self.weights = [] 
        self.FL_output = []
        self.biases = np.ones(output_dim) 
        self.alpha = 0.001
        self.flattened_input = []


    def forward_propagation(self, flatten): 
        print("Fully Connected ",end=" ")
        self.flattened_input = flatten
        self.weights = np.random.randint(0,10, size=(self.output_dim, flatten.shape[1]))/ flatten.shape[1]
        FL_output = []
        for i in range(flatten.shape[0]): 
            FL_output.append(np.tanh(np.dot(self.weights, (flatten[i])) + self.biases))
        self.FL_output = np.array(FL_output)
        return self.FL_output

    def back_propagation(self, del_Z): 
        print("Fully Connected ",end=" ")  
        del_W = np.dot(self.flattened_input.T, del_Z) / del_Z.shape[0]
        del_b = np.sum(del_Z, axis=0)/del_Z.shape[0]
        del_f = np.dot(del_Z, self.weights)

        self.update_parameters(del_W.T, del_b)

        return del_f 

    def update_parameters(self, delW, delB):
        self.biases = self.biases - (delB*self.alpha)
        self.weights = self.weights - (delW*self.alpha)
    


class Softmax:
    def __init__(self):
        self.input = []
        self.output = []

    def one_hot_encoding(self, labels): 
        num_classes = 10 
        one_hot_labels = np.zeros((len(labels), num_classes)) 
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1

        return one_hot_labels

    
    def forward_propagation(self, a_out): 
        print("Softmax ",end=" ")
        self.input = a_out
        output = [] 
        for i in range(len(a_out)):
            y_pred = np.exp(a_out[i])
            y_pred = y_pred/np.sum(y_pred)
            output.append(np.exp(a_out[i])/np.sum(np.exp(a_out[i])))
        self.output = np.array(output)
        return self.output

    def back_propagation(self, y):
        print("Softmax ",end=" ")
        y_pred = self.one_hot_encoding(labels= y)
        del_Z = np.array(self.output) - np.array(y_pred) 
        return del_Z
         


