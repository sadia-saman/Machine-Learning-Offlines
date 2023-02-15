from sklearn.metrics import confusion_matrix
import time
import sys
import pickle as pkl
import cv2
import os
import pandas as pd
import numpy as np

class Layer:
    def forward_propagation(self, input):
        # Define the forward pass for the layer
        pass
        
    def backward_propagation(self, input):
        # Define the backward pass for the layer
        pass


class Convolution(Layer) :
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
        self.biases = np.ones(N_out_channel) 
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
        #print("Convolution ",end=" ")

        self.X = np.empty((np.shape(images)[0], np.shape(images)[1]), dtype=object)
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
        #print("Convolution ",end=" ")
        del_K = np.zeros(self.kernels.shape) 

        for idx in range(del_K.shape[0]):
            for k in range(np.shape(self.X)[0]): 
                for i in range(np.shape(self.X)[1]):
                    image = self.X[k][i]
                    del_K[idx] += self.convolve(image, del_Z[k][idx], del_K.shape[1], del_K.shape[2]) 

        del_B = np.sum(np.sum(np.sum(del_Z, axis=3),axis = 2), axis=0)

        rotated_kernels = np.array([np.fliplr(np.flipud(kernel)) for kernel in self.kernels]) 
        del_X = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], np.shape(self.X[0][0])[0], np.shape(self.X[0][0])[1]))

        padding_x =((np.shape(del_Z)[2]*self.stride) - np.shape(del_X)[2]) + self.filter_dimension - self.stride - self.padding
        padding_y =((np.shape(del_Z)[3]*self.stride) - np.shape(del_X)[3]) + self.filter_dimension - self.stride - self.padding
        

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

        

class ReLU(Layer):
    def __init__(self):
        self.RelU_input = []
        self.shape = None
    def forward_propagation(self, feature_map): 
        #print("ReLU ",end=" ")
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
        #print("ReLU ",end=" ")
        del_C_by_del_Z = np.zeros(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for x in range(self.shape[2]):
                    for y in range(self.shape[3]):
                        if self.ReLU_input[i,j,x,y] > 0:
                            del_C_by_del_Z[i,j,x,y] = 1
        return del_C_by_del_Z * del_C 
        


class Pooling(Layer):
    def __init__(self, pool_dimension, stride):
        self.pool_dimension = pool_dimension
        self.stride = stride
        self.pool_input = []
        self.shape = None 
    
    def forward_propagation(self, feature_map): 
        #print("Pooling ",end=" ")
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
        #print("Pooling ",end=" ")
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
        

class FlattenLayer(Layer):
    def __init__(self):
        self.pool_shape = None

    
    def normalize(self, x):
        x_hat = (x - np.mean(x)) / np.std(x)
        return x_hat

    def forward_propagation(self, pool): 
        #print("Flatten ",end=" ")
        self.pool_shape = pool.shape
        pool = np.array(pool)
        flattened_vector = []
        for img_idx in range(pool.shape[0]):
            flattened_vector.append(self.normalize(pool[img_idx].flatten())) 

        return np.array(flattened_vector)

    def back_propagation(self, del_Z): 
        #print("Flatten ",end=" ")
        del_f =  np.reshape(del_Z, self.pool_shape)
        return del_f

class FullyConnectedNN(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim 
        self.weights = []  
        self.biases = [] 
        self.alpha = 0.001
        self.flattened_input = []


    def forward_propagation(self, flatten): 
        #print("Fully Connected ",end=" ")
        self.flattened_input = flatten
        if self.weights == []:
            self.weights = np.random.randint(0,10, size=(self.output_dim, flatten.shape[1]))/ flatten.shape[1]
        if self.biases == []:
            self.biases = np.ones(self.output_dim)

        FL_output = []
        for i in range(flatten.shape[0]): 
            FL_output.append(np.tanh(np.dot(self.weights, (flatten[i])) + self.biases)) 

        return np.array(FL_output)

    def back_propagation(self, del_Z): 
        #print("Fully Connected ",end=" ")  
        del_W = np.dot(self.flattened_input.T, del_Z) / del_Z.shape[0]
        del_b = np.sum(del_Z, axis=0)/del_Z.shape[0]
        del_f = np.dot(del_Z, self.weights)

        self.update_parameters(del_W.T, del_b)

        return del_f 

    def update_parameters(self, delW, delB):
        self.biases = self.biases - (delB*self.alpha)
        self.weights = self.weights - (delW*self.alpha)
    


class Softmax(Layer):
    def __init__(self): 
        pass
    
    def forward_propagation(self, a_out): 
        output = [] 
        for i in range(len(a_out)): 
            output.append(np.exp(a_out[i])/np.sum(np.exp(a_out[i]))) 
        return np.array(output)

    def back_propagation(self, del_Z):  
        self.delZ = del_Z 
        return del_Z
         

class Model:
    def __init__(self): 
        self.components = [] 

    def one_hot_encoding(self, labels): 
        num_classes = 10 
        one_hot_labels = np.zeros((len(labels), num_classes)) 
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1

        return one_hot_labels

    def Load_Data(self):

        data_folder_name = sys.argv[1]
        data_folder_path = data_folder_name + "/Coding/" 
        csv_file = data_folder_name+"/Coding.csv"

        images = []
        labels = []    

        df = pd.read_csv(csv_file)
        labels = df['digit'].values

        print("Loading data from ", data_folder_path)

        for file in os.listdir(data_folder_path):
            img = cv2.imread(data_folder_path + file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                img = cv2.resize(img, (80,80))
                l = []
                l.append(np.array(img, dtype=np.float32)/255)
                images.append(l)

        print("Dataset dimension ", np.shape(images),"==>",np.shape(images[0]), np.shape(labels))

        images = np.array(images) 
        labels = np.array(self.one_hot_encoding(labels)) 
        

        validation_size = (int) (images.shape[0]*0.25)

        validation_images = images[0:validation_size , : , : , :]
        validation_labels = labels[0:validation_size]

        train_images = images[validation_size: , : , : , :]
        train_labels = labels[validation_size:]
        return train_images, train_labels, validation_images, validation_labels

    def taking_commands(self):
        with open('model.txt','r') as file:
            commands = file.read().split("\n") 
        for command in commands : 
            command_tokens = command.split(" ")
            if command_tokens[0]== "Conv":
                N_output_channels = (int)(command_tokens[1])
                Filter_dimension  = (int)(command_tokens[2])
                Stride            = (int)(command_tokens[3])
                Padding           = (int)(command_tokens[4])
                Conv = Convolution(N_output_channels, Filter_dimension, Stride, Padding) 
                self.components.append(Conv)
            elif command_tokens[0]== "ReLU":
                ReLU_ = ReLU()
                self.components.append(ReLU_)
            elif command_tokens[0]== "Pool":
                pool_dim = (int)(command_tokens[1])
                stride  = (int)(command_tokens[2])
                Pool = Pooling(pool_dim, stride)
                self.components.append(Pool)
            elif command_tokens[0]== "Flatten":
                fl = FlattenLayer()
                self.components.append(fl)
            elif command_tokens[0]== "FC":
                N_output_channels = (int)(command_tokens[1])
                FC_nn = FullyConnectedNN(N_output_channels)
                self.components.append(FC_nn)
            elif command_tokens[0]== "Softmax":
                softmax = Softmax()
                self.components.append(softmax)
        print("components size  ", np.shape(self.components))


    def train(self, data, labels ): 
        i = 0
        forward_in = data
        for component in self.components: 
            forward_in = component.forward_propagation(forward_in) 

        backward_in = labels - forward_in 

        for component in reversed(self.components): 
            backward_in = component.back_propagation(backward_in) 
        return forward_in
    
    def predict(self, data):
        forward_in = data
        for component in self.components:
            forward_in = component.forward_propagation(forward_in)
        return forward_in

    def cross_entropy_loss(self, labels, predictions):
        return -np.sum(labels*np.log(predictions)) 

    def f1_score(self, labels, predictions): 
        confusion = confusion_matrix(labels, predictions, labels=range(10))
        TP = np.zeros(10)
        TN = np.zeros(10)
        FN = np.zeros(10)
        
        for i in range(10):
            TP[i] = confusion[i][i]
            for j in range(10):
                if i != j:
                    TN[i] += confusion[j][j]
                    FN[i] += confusion[i][j] - confusion[i][i]
        accuracy = np.sum(TP)/(np.sum(TP)+np.sum(FN))
        f1_score = np.sum(2*TP)/(2*np.sum(TP)+np.sum(FN)+np.sum(TN))                
        return f1_score, accuracy

    
    def fit(self, BATCH_SIZE, EPOCHS, train_X, train_y, validation_X, validation_y ):
        EPOCHS = (int)(EPOCHS)
        accuracy = np.empty(EPOCHS)
        f1_score = np.empty(EPOCHS)
        for epoch in range(EPOCHS):
            print("Epoch Running................................................ ", epoch)
            y_pred = np.zeros((train_X.shape[0], 10))
            for i in range(0, train_X.shape[0], BATCH_SIZE):
                print("Batch Running................................................ ", i)
                data = train_X[i:i+BATCH_SIZE, : , : , :]
                labels = train_y[i:i+BATCH_SIZE]
                predictions_training = self.train(data, labels) 
                y_pred[i:i+BATCH_SIZE] = predictions_training
        
            training_loss = self.cross_entropy_loss(train_y, y_pred)
            labels_array = np.argmax(train_y,axis=1)
            predictions_array = np.argmax(y_pred,axis=1)
            training_F1_score, accuracy = self.f1_score(labels_array, predictions_array)

            #validation..........
            validation_y_pred = self.predict(validation_X)
            validation_loss = self.cross_entropy_loss(validation_y, validation_y_pred)

            validation_labels_array = np.argmax(validation_y, axis=1)
            validation_predictions_array = np.argmax(validation_y_pred, axis=1)
            validation_F1_score, validation_accuracy = self.f1_score(validation_labels_array, validation_predictions_array)

            epoch =(int)(epoch)
            accuracy[epoch] = validation_accuracy
            f1_score[epoch] = validation_F1_score

            self. prev_acc = 0
            self.prev_f1 = 0

            if epoch > 0:
                if self.prev_acc <= validation_accuracy or self.prev_f1 <= validation_F1_score:
                    pkl.dump(self, "1705102_model.pickle") 
            self.prev_acc = validation_accuracy
            self.prev_f1 = validation_F1_score 
            
            print("Epoch = ", epoch, "Training Loss = ", training_loss, "Training F1_score = ", training_F1_score, "Training Accuracy = ", accuracy)
            print("Epoch = ", epoch, "Validation Loss = ", validation_loss, "Validation F1_score = ", validation_F1_score, "Validation Accuracy = ", validation_accuracy)

    def test(self, images, filename):
        test_y_pred = self.predict(images)
        """ test_loss = self.cross_entropy_loss(labels, test_y_pred)
        test_labels_array = np.argmax(labels,axis=1) """
        test_predictions_array = np.argmax(test_y_pred,axis=1)
        """ test_F1_score, test_accuracy = self.f1_score(test_labels_array, test_predictions_array)
        print("Test Loss = ", test_loss, "Test F1_score = ", test_F1_score, "Test Accuracy = ", test_accuracy)
        """
        with open("17051705102_prediction.csv", 'w') as f:
            f.write("FileName","Digit")
            for i in range(len(test_predictions_array)):
                f.write(filename[i], test_predictions_array[i])
        
        """ return test_loss, test_accuracy, test_F1_score """



if __name__=="__main__":
    model = Model()

    start_time = time.time()
    train_X, train_y, validation_X, validation_y = model.Load_Data()
    print("Time taken to load data = ", time.time() - start_time)

    start_time = time.time()
    model.taking_commands()
    print("Time taken to take commands = ", time.time() - start_time)

    start_time = time.time()
    BATCH_SIZE = 32
    EPOCHS = 5
    model.fit(BATCH_SIZE, EPOCHS, train_X, train_y, validation_X, validation_y)
    print("Time taken to train = ", time.time() - start_time)  




    
