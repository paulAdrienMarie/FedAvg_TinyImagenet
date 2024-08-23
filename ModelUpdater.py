import os
import onnx
import numpy as np
from onnx import numpy_helper

# Directory setup
model_path = os.path.join(os.path.dirname(__file__), "model", "inference.onnx")

class ModelUpdater:
    """
    Stores and updates weights of the model
    
    Attributs:
    model_path -- Path to the file in which to save the updated model
    nb_roc -- Number of client - server communication 
    nb_users -- Number of users during the FL simulation
    classifier_weights -- List to store the udpated weights of the weights of the classification head
    classifier_bias -- List to store the udpated bias of the bias of the classification head
    """
    
    def __init__(self, model_path, nb_users,nb_roc):
        """
        Initializes a new instance of the ModelUpdater class
        
        Arguments:
        updated_model_path -- Path to the file in which to save the updated model
        nb_roc -- Number of round of communications
        nb_users -- Number of users in the simulation
        """
        
        self.model_path = model_path
        self.nb_roc = nb_roc
        self.nb_users = nb_users
        self.classifier_weights = []
        self.classifier_bias = []
    
    def update_weights(self, updated_weights):
        """Store the weights from a client for averaging later."""
        classifier_weight_array = np.array(updated_weights[:153600], dtype=np.float32).reshape(200, 768)
        classifier_bias_array = np.array(updated_weights[153600:], dtype=np.float32).reshape(200, )
        
        self.classifier_weights.append(classifier_weight_array)
        self.classifier_bias.append(classifier_bias_array)
        
        print(f"Received data from {len(self.classifier_weights)} clients out of {self.nb_users} clients")

    def average_parameters(self, parameters_list):
        """Average the parameters collected from all clients."""
        if len(parameters_list) == 0:
            return None
        
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
        if not self.fc1_weights:
            print("No user data to process")
            return {"message": "No user data to process"}
        
        print("Loading the model")
        model = onnx.load(self.model_path)
        
        # Average the parameters
        print("Start averaging the parameters")
        classifier_weight_avg = self.average_parameters(self.classifier_weights)
        classifier_bias_avg = self.average_parameters(self.classifier_bias)
        
        # Update the model with the averaged parameters
        print("Start updating the model parameters")
        self.copy_to_model(model, "classifier.weight", classifier_weight_avg)
        self.copy_to_model(model, "classifier.bias", classifier_bias_avg)
        
        print("Saving the model")
        onnx.save_model(model, self.model_path)
        print("Model saved successfully.")
        print("Emptying the weights")
        self.reset()
        return {"message": "Model updated with averaged parameters"}

    def reset(self):
        """Clear all accumulated weights and biases."""
        self.classifier_weights = []
        self.classifier_bias = []
        print("Reset all weight and bias lists.")
        
# Example usage
if __name__ == "__main__":
    updater = ModelUpdater(model_path=model_path)
    # Example: Pass the weights collected from clients to `update_weights` method
    # updated_weights = ... (collected from clients)
    # updater.update_weights(updated_weights)
    updater.update_model()
