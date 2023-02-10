from CNN_1705102 import *

class Model:
    def __init__(self):
        self.feature_map = []
        self.images = []
        self.labels = []
        self.components = []

    def Load_Data(self):

        data_folder_name = input("Enter the name of the folder containing the data: ")
        data_folder_path = "../data/" + data_folder_name + "/" 
        csv_file = "../data/" + data_folder_name+".csv"

        images = []
        labels = []   

        df = pd.read_csv(csv_file)
        labels = df['digit'].values

        for file in os.listdir(data_folder_path):
            img = cv2.imread(data_folder_path + file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                l = []
                l.append(np.array(img, dtype=np.float32)/255)
                images.append(l)

        print("Dataset dimension ", np.shape(images),"==>",np.shape(images[0]), np.shape(labels))
        self.images = np.array(images) 
        self.labels = np.array(labels) 

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


    def train(self):
        forward_in = self.images
        i = 0
        for component in self.components:
            print(i, end=" ")
            i += 1
            print("forward_in shape = ",np.shape(forward_in), end=" ")
            forward_in = component.forward_propagation(forward_in)
            print("forward_out shape = ",np.shape(forward_in))

        backward_in = self.labels

        for component in reversed(self.components):
            i -= 1
            print(i, end=" ")
            print("backward_in = ",np.shape(backward_in), end=" ")
            backward_in = component.back_propagation(backward_in)
            print("backward_out = ",np.shape(backward_in))

if __name__=="__main__":
    model = Model()
    model.Load_Data()
    model.taking_commands()
    model.train()