from CNN_1705102 import *
from sklearn.metrics import confusion_matrix
import time
import sys

class Model:
    def __init__(self):
        self.feature_map = []
        self.images = []
        self.labels = []
        self.components = []
        self.predictions_training = []

    def one_hot_encoding(self, labels): 
        num_classes = 10 
        one_hot_labels = np.zeros((len(labels), num_classes)) 
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1

        return one_hot_labels

    def Load_Data(self):

        data_folder_name = sys.argv[1]
        data_folder_path = data_folder_name + "/training-b/" 
        csv_file = data_folder_name+"training-b.csv"

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

        self.images = np.array(images) 
        self.labels = np.array(self.one_hot_encoding(labels)) 
        

        validation_size = (int) (self.images.shape[0]*0.25)

        self.validation_images = self.images[0:validation_size , : , : , :]
        self.validation_labels = self.labels[0:validation_size]

        self.images = self.images[validation_size: , : , : , :]
        self.labels = self.labels[validation_size:]

    def taking_commands(self):
        with open('command.txt','r') as file:
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
            """ print(i, end=" ")
            i += 1
            print("forward_in shape = ",np.shape(forward_in), end=" ") """
            forward_in = component.forward_propagation(forward_in)
            """ print("forward_out shape = ",np.shape(forward_in)) """

        backward_in = labels
        self.predictions_training = np.array(forward_in)

        for component in reversed(self.components):
            """ i -= 1
            print(i, end=" ")
            print("backward_in = ",np.shape(backward_in), end=" ") """
            backward_in = component.back_propagation(backward_in)
            """ print("backward_out = ",np.shape(backward_in)) """
    
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

    
    def fit(self, BATCH_SIZE, EPOCHS):
        for epoch in range(EPOCHS):
            print("Epoch Running................................................ ", epoch)
            y_pred = np.zeros((self.images.shape[0], 10))
            for i in range(0, self.images.shape[0], BATCH_SIZE):
                data = self.images[i:i+BATCH_SIZE, : , : , :]
                labels = self.labels[i:i+BATCH_SIZE]
                self.train(data, labels) 
                y_pred[i:i+BATCH_SIZE] = self.predictions_training
        
            training_loss = self.cross_entropy_loss(self.labels, y_pred)
            labels_array = np.argmax(self.labels,axis=1)
            predictions_array = np.argmax(y_pred,axis=1)
            training_F1_score, accuracy = self.f1_score(labels_array, predictions_array)

            #validation..........
            validation_y_pred = self.predict(self.validation_images)
            validation_loss = self.cross_entropy_loss(self.validation_labels, validation_y_pred)
            validation_labels_array = np.argmax(self.validation_labels,axis=1)
            validation_predictions_array = np.argmax(validation_y_pred,axis=1)
            validation_F1_score, validation_accuracy = self.f1_score(validation_labels_array, validation_predictions_array)
    
            print("Epoch = ", epoch, "Training Loss = ", training_loss, "Training F1_score = ", training_F1_score, "Training Accuracy = ", accuracy)
            print("Epoch = ", epoch, "Validation Loss = ", validation_loss, "Validation F1_score = ", validation_F1_score, "Validation Accuracy = ", validation_accuracy)

    def test(self):
        test_y_pred = self.predict(self.test_images)
        test_loss = self.cross_entropy_loss(self.test_labels, test_y_pred)
        test_labels_array = np.argmax(self.test_labels,axis=1)
        test_predictions_array = np.argmax(test_y_pred,axis=1)
        test_F1_score, test_accuracy = self.f1_score(test_labels_array, test_predictions_array)
        print("Test Loss = ", test_loss, "Test F1_score = ", test_F1_score, "Test Accuracy = ", test_accuracy)


if __name__=="__main__":
    model = Model()

    start_time = time.time()
    model.Load_Data()
    print("Time taken to load data = ", time.time() - start_time)

    start_time = time.time()
    model.taking_commands()
    print("Time taken to take commands = ", time.time() - start_time)

    start_time = time.time()
    BATCH_SIZE = 64
    EPOCHS = 5
    model.fit(BATCH_SIZE, EPOCHS)
    print("Time taken to train = ", time.time() - start_time) 

    start_time = time.time()
    model.test()
    print("Time taken to test = ", time.time() - start_time)


    
