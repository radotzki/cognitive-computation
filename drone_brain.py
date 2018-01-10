import numpy as np
import math

#Hyper Params
hidden_dim = 8
input_dim = 8
num_hidden_layers= 2
output_dim = 2
SMALL_DISTRIBUTION = 0.01 # spread of gaussian for starting random values

def sigmoid(x):
    """
    Simple Sigmoid Activation Function
    """
    return 1 / (1 + math.exp(-x))

lambda_sigmoid = np.vectorize(sigmoid) #map function for np arrays of sigmoid

def sigmoid_derivative(x):
    return x * (1.0 - x)
lambda_sigmoid_derivitive = np.vectorize(sigmoid_derivative) # mapping function for np arrays


class neural_network:
    """
    Sigmoid activation function on hidden layers
    Tanh activation function on output layer
    """

    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        self.neurons = []

        #add weights from input to first hidden layer
        self.layers.append(np.random.normal(size=(input_dim, hidden_dim)))

        #add weights of each hidden layer to adjacent hidden layer
        for _ in range(num_hidden_layers - 1):
            self.layers.append(np.random.normal(scale=SMALL_DISTRIBUTION, size=(hidden_dim, hidden_dim)))
            self.neurons.append(np.empty(hidden_dim))

        #add weights from last hidden layer to output layer
        self.layers.append(np.random.normal(scale=SMALL_DISTRIBUTION, size=(hidden_dim, output_dim)))
        self.neurons.append(np.empty(hidden_dim))

    def get_result(self, prev_frame, drone_x, drone_y, obj_x, obj_y):
        prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y = prev_frame
        input_layer = np.array([prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
                    drone_x, drone_y, obj_x, obj_y])


        #input to first hidden layer
        h1_result = np.dot( input_layer, self.layers[0]) # mat_mult dimensionality: input * [input * hidden] = hidden
        self.neurons[0] = lambda_sigmoid(h1_result)#apply sigmoid activation function


        #hidden layer to hidden layer
        prev_hidden_result = self.neurons[0]
        for layer in range(self.num_hidden_layers):
            # mat_mult dimensionality: hidden * [hidden * hidden] = hidden
            prev_hidden_result = np.dot(prev_hidden_result, self.layers[layer])
            self.neurons[layer] = lambda_sigmoid(prev_hidden_result)

        #final hidden layer to output layer
        ouput_layer_result = np.dot(prev_hidden_result, self.layers[layer + 1]) # mat_mult dimensionality: hidden * [hidden * ouput] = ouput

        # Tanh activation function on output
        final_result = np.tanh(ouput_layer_result, out=None)
        x_move, y_move = final_result

        self.update_weights(prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
                            drone_x, drone_y, obj_x, obj_y, x_move, y_move)
        return x_move, y_move


    def update_weights(self, prev_drone_x, prev_drone_y, prev_obj_x, prev_obj_y,
                       drone_x, drone_y, obj_x, obj_y, x_move, y_move):

        # Based on heuristic guess of a good move, imitating a reward function
        x_error, y_error = self.calc_error(prev_obj_x, prev_obj_y, obj_x, obj_y, prev_drone_x, prev_drone_y,
                                          x_move, y_move)

        #Calculate error per neuron
        previous_layer_ideal = [x_move, y_move]
        for layer in reversed(range(len(self.layers))):

            cur_layer = self.layers[layer]
            print(str(layer) + ' layer')

            if layer == len(self.layers)-1: #final layer
                error = [x_error, y_error]
                error_per_neuron = np.dot(cur_layer, error) # weights * error
                error_per_weights = cur_layer * error
                previous_layer_ideal = self.neurons[layer - 1] - error_per_neuron

            else: #all other layers
                error = previous_layer_ideal - self.neurons[layer]
                error_per_weights = cur_layer * error
                error_per_neuron = np.dot(cur_layer, error) # weights * error
                previous_layer_ideal = self.neurons[layer] - error_per_neuron
            # Gradient Descent step
            self.layers[layer] = self.layers[layer] - (alpha * error_per_weights)


    def calc_error(self, prev_obj_x, prev_obj_y, obj_x, obj_y, prev_drone_x, prev_drone_y,
                   actual_x_move, actual_y_move):
        #simple heuristic chosen for this experiment as to an ideal position for the drone to aim for
        ideal_x_pos = (prev_obj_x + obj_x) / 2
        ideal_y_pos = (prev_obj_y + obj_y) / 2

        # ideal move needed to reach ideal location
        ideal_x_move = ideal_x_pos - prev_drone_x
        ideal_y_move = ideal_y_pos - prev_drone_y

        #error from 'best choice'
        x_error = abs(ideal_x_move - actual_x_move)
        y_error = abs(ideal_y_move - actual_y_move)

        return x_error, y_error



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
        #update previous frame
        self.prev_frame = drone_x, drone_y, obj_x, obj_y

        return result