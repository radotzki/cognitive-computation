import numpy as np
import math

# input variables
alpha = 0.1
input_dim = 8
hidden_dim = 8
num_hidden_layers= 2
output_dim = 2

def sigmoid(x):
    """
    Simple Sigmoid Activation Function
    """
    return 1 / (1 + math.exp(-x))

lambda_sigmoid = np.vectorize(sigmoid) #map function for np arrays of sigmoid

def sigmoid_derivative(x):
    return x * (1.0 - x)

class neural_network:
    
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        
        #add weights from input to first hidden layer
        self.layers.append(np.random.normal(size=(input_dim, hidden_dim)))
        
        #add weights of each hidden layer to adjacent hidden layer
        for _ in range(num_hidden_layers - 1):
            self.layers.append(np.random.normal(size=(hidden_dim, hidden_dim)))
            
        #add weights from last hidden layer to output layer
        self.layers.append(np.random.normal(size=(hidden_dim, output_dim)))
    
    def get_result(self, prev_frame, drone_x, drone_y, obj_x, obj_y):
        prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y = prev_frame
        input_layer = np.array([prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
                    drone_x, drone_y, obj_x, obj_y])
        
        
        #input to first hidden layer
        h1_result = np.dot( input_layer, self.layers[0]) # mat_mult dimensionality: input * [input * hidden] = hidden 
        h1_result = lambda_sigmoid(h1_result)#apply sigmoid activation function
        

        #hidden layer to hidden layer
        last_hidden_result = h1_result
        for layer in range(self.num_hidden_layers):
            # mat_mult dimensionality: hidden * [hidden * hidden] = hidden
            last_hidden_result = np.dot(last_hidden_result, self.layers[layer])
            last_hidden_result = lambda_sigmoid(last_hidden_result)
        
        #final hidden layer to output layer
        ouput_layer_result = np.dot(last_hidden_result, self.layers[layer + 1]) # mat_mult dimensionality: hidden * [hidden * ouput] = ouput 
        final_result = np.tanh(ouput_layer_result, out=None)
        x_move, y_move = final_result
        
        #self.update_weights(prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
        #                    drone_x, drone_y, obj_x, obj_y, x_move, y_move)
        return x_move, y_move
                           
    
    def update_weights(self, prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
                       drone_x, drone_y, obj_x, obj_y, x_move, y_move):
        
        # Based on heuristic guess of a good move, imitating a reward function
        x_error, y_error = self.calc_error(prev_obj_x, prev_obj_y, obj_x, obj_y, prev_drone_x, prev_drone_y)
        
        #Calculate error t
        
        
        
    def calc_error(self, prev_obj_x, prev_obj_y, obj_x, obj_y, prev_drone_x, prev_done_y):
        #simple heuristic chosen for this experiment as to an ideal position for the drone to aim for
        ideal_x_pos_chosen = (prev_obj_x + obj_x) / 2
        ideal_y_pos_chosen = (prev_obj_y + obj_y) / 2
        
        # ideal move needed to reach ideal location
        ideal_x_move = ideal_x_pos - prev_drone_x
        ideal_y_move = ideal_y_pos - prev_drone_y
        
        #error from 'best choice'
        x_error = abs(ideal_x_move - x_move)
        y_error = abs(ideal_y_move - y_move)
        
        return x_error,y_error
            
        

class drone_brain:
    def __init__(self, start_drone_x, start_drone_y, start_obj_x, start_obj_y):
        self.prev_frame = start_drone_x, start_drone_y, start_obj_x, start_obj_y
        self.ANN = neural_network(num_hidden_layers, input_dim, hidden_dim, output_dim)
    
    def get_move(self, drone_x, drone_y, obj_x, obj_y):
        """
        a_x, a_y = x,y coordinates of drone
        b_x, b_y = x,y coordinates of object
        """
        result = self.ANN.get_result(self.prev_frame, drone_x, drone_y, obj_x, obj_y)
        print(result)
        #update previous frame
        self.prev_frame = drone_x, drone_y, obj_x, obj_y

        #return x_prime, y_prime