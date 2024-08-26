import os
import onnx
import numpy as np
from onnx import numpy_helper

# Directory setup
model_path = os.path.join(os.path.dirname(__file__), "model", "inference.onnx")

class ModelUpdater:
    """
    Stores and updates weights of the model
    
    Attributes:
    model_path -- Path to the file in which to save the updated model
    nb_roc -- Number of client-server communications 
    nb_users -- Number of users during the FL simulation
    classifier_weights -- List to store the updated weights of the classification head
    classifier_bias -- List to store the updated bias of the classification head
    """
    
    def __init__(self, model_path, nb_users, nb_roc):
        """
        Initializes a new instance of the ModelUpdater class
        
        Arguments:
        model_path -- Path to the file in which to save the updated model
        nb_roc -- Number of rounds of communications
        nb_users -- Number of users in the simulation
        """
        
        self.model_path = model_path
        self.nb_roc = nb_roc
        self.nb_users = nb_users
        self.parameters = {
            "classifier.weight": [],
            "classifier.bias": []
        }
        self.model = onnx.load(self.model_path)
    
    def update_weights(self, updated_weights):
        """Store the weights from a client for averaging later."""
        index = 0
        for param in self.parameters.keys():
            shape = self.get_parameters_shape(param)
            num_elements = np.prod(shape)
            self.parameters[param].append(np.array(updated_weights[index:index + num_elements], dtype=np.float32).reshape(shape))
            index += num_elements
        
        print(f"Received data from {len(self.parameters['classifier.weight'])} clients out of {self.nb_users} clients")
        
    def get_parameters_shape(self, name):
        for initializer in self.model.graph.initializer:
            if initializer.name == name:
                # The shape of the initializer (i.e., the parameter)
                shape = initializer.dims
                return tuple(shape)  # Return the shape as a tuple
        # If the parameter with the given name is not found, return None or raise an exception
        return None

    def average_parameters(self, parameters_list):
        """Average the parameters collected from all clients."""
        if len(parameters_list) == 0:
            return None
        print("Averaging models parameters")
        return np.mean(parameters_list, axis=0).astype(np.float32)

    def copy_to_model(self, model, name, params):
        """Copy the averaged parameters to the ONNX model."""
        print(f"Updating parameters of {name}")
        for initializer in model.graph.initializer:
            if initializer.name == name:
                new_weights_tensor = numpy_helper.from_array(params, name=initializer.name)
                initializer.CopyFrom(new_weights_tensor)

    def update_model(self):
        """Update the ONNX model with the averaged parameters."""
        if not self.parameters["classifier.weight"] or not self.parameters["classifier.bias"]:
            print("No user data to process")
            return {"message": "No user data to process"}
        
        print("Loading the model")
        self.model = onnx.load(self.model_path)
        
        # Average the parameters
        print("Start averaging the parameters")
        for param_name in self.parameters:
            self.copy_to_model(
                self.model,
                param_name, 
                self.average_parameters(self.parameters[param_name])
            )
        
        print("Saving the model")
        onnx.save_model(self.model, self.model_path)
        print("Model saved successfully.")
        print("Emptying the weights")
        self.reset()
        return {"message": "Model updated with averaged parameters"}

    def reset(self):
        """Clear all accumulated weights and biases."""
        self.parameters = {
            "classifier.weight": [],
            "classifier.bias": []
        }
        print("Reset all weight and bias lists.")
        

