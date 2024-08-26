import platform
import onnx
from onnxruntime.training import artifacts
import os
import torch
from transformers import ViTForImageClassification
import torch.nn as nn
import argparse
assert list(platform.python_version_tuple())[:-1] == ["3", "9"]

class Model(torch.nn.Module):
    """
    Class for the modified version of google/vit-base-patch16-224
    
    Attributs:
    nb_classes -- Number of classes of the modified classification head, 
                  Implecitly equals to 200 
    model_path -- Relatvie path where to save the model
    """
    
    def __init__(self, nb_classes, model_path):
        """Initializes a new instance of the Model class"""
        super(Model, self).__init__()
        self.nb_classes = nb_classes
        self.model_name = 'google/vit-base-patch16-224'
        self.model_path = model_path
        
    def load_model(self):
        """Loads the pre-trained ViT model."""
        return ViTForImageClassification.from_pretrained(self.model_name)
    
    def modify_layer(self):
        """Modify the classification head to match the number of classes."""
        model = self.load_model()
        model.classifier = nn.Linear(model.classifier.in_features, self.nb_classes)
        return model
    
    def save_to_onnx(self):
        """Saves the resulting model to an ONNX file."""
        model = self.modify_layer()
        model.eval()  # Set the model to evaluation mode
        
        input_names = ["pixel_values"]
        output_names = ["output"]
        
        # Specify dynamic axes for batch size, height, and width
        dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "output": {0: "batch_size"}
        }
        
        # Example input tensor with a flexible size
        dummy_input = torch.randn(1, 3, 224, 224)  # Initial size, but ONNX will allow dynamic sizes
        
        # Export the model to ONNX
        torch.onnx.export(
            model,
            dummy_input,  # This should be an example input tensor
            self.model_path,  # Path to the file where the ONNX model will be saved
            input_names=input_names,
            output_names=output_names,
            opset_version=14,
            do_constant_folding=False,  # Optimization during export
            training=torch.onnx.TrainingMode.TRAINING,  # Use EVAL mode for export
            dynamic_axes=dynamic_axes,  # Allow dynamic sizes for batch, height, and width
            export_params=True,
            keep_initializers_as_inputs=False
        )
        
    def __call__(self):
        """Call this method to export the model to ONNX format."""
        self.save_to_onnx()
    

class Artifacts: 
    """
    Class to handle the generation of the training artifacts
    
    Attributs:
    model_path -- Path to the onnx model
    artifacts_path -- Path where to save the artifacts
    """
    
    def __init__(self, model_path, artifacts_path, nb_classes):
        """Initializes a new instance of the Artifacts class"""
        self.model_path = model_path
        self.artifacts_path = artifacts_path
        self.model = Model(
            nb_classes=nb_classes,
            model_path=self.model_path
        )
        
    def create_model(self):
        """Creates and exports the model"""
        self.model()
        
    def load_model(self):
        """Loads and returns the model"""
        onnx_model = onnx.load_model(self.model_path)
        assert self.check_model(onnx_model)
        return onnx_model
        
        
    def check_model(self,model):
        """
        Checks wether the model is valid or not
        
        Arguments:
        model -- The onnx model to check]
        """
        
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: {}".format(e))
        else:
            print("The exported model is valid!")
            return True
        
    def gen_artifacts(self):
        """Generates the training artifacts"""
        onnx_model = self.load_model()
        
        # Only train the classifier head 
        requires_grad = ["classifier.weight","classifier.bias"]
        frozen_params = [
            param.name
            for param in onnx_model.graph.initializer
            if param.name not in requires_grad
        ]
        
        output_names = ["output"]
                
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.AdamW,
            loss=artifacts.LossType.L1Loss,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            additional_output_names=output_names,
            artifact_directory=os.path.join(os.path.dirname(__file__), "artifacts")
        )
        
    def __call__(self):
        """Generates training artifacts"""
        self.create_model()
        self.gen_artifacts()

if __name__=="__main__":
    
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path = os.path.join(os.path.dirname(__file__), "onnx/inference.onnx")
    
    parser = argparse.ArgumentParser(
        prog="Create artifacts"
    )
    
    parser.add_argument("--nb_classes", type=int, help="Number of classes of the model")
    
    args = parser.parse_args()
    
    obj = Artifacts(
        model_path=model_path,
        artifacts_path=artifacts_path,
        nb_classes=args.nb_classes
    )
    obj()
    
    

