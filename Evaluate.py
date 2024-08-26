from torchvision import transforms
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.training.api import CheckpointState, Module
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
import evaluate

class Test:
    """
    Tests the updated model after FedAvg
    
    Attributes:
    path_to_model -- Path to the model to test
    metrics -- Dictionary to store the metrics over the communication rounds
    """
    
    def __init__(self):
        """Initializes a new instance of the Test class"""
        self.path_to_model = "./onnx/inference.onnx"
        self.metrics = {
            "accuracies": [],
            "losses": []
        }
   
    def collate_fn(self, batch):
        images = []
        labels = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        for item in batch:
            image = transform(item['image'])
            label = item['label']
            images.append(image)
            labels.append(label)
        
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels
   
    def load_test_images(self):
        batch_size = 64
        tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='valid')  # Use the same dataset
        test_loader = DataLoader(tiny_imagenet, batch_size=batch_size, collate_fn=self.collate_fn)
        return test_loader
    
    def load_inference_session(self):
        """Loads Inference Session"""
        return ort.InferenceSession(self.path_to_model)
    
    def softmax_activation(self, batch_logits):
        return torch.tensor(np.argmax(batch_logits, axis=-1))

    def generate_target_logits(self, batch_target):
        batch_size = len(batch_target)
        num_classes = 200  # Assuming 200 classes for tiny-imagenet
        target_logits = torch.zeros([batch_size, num_classes], dtype=torch.float32)

        for i, target in enumerate(batch_target):
            target_logits[i, target] = 1

        return target_logits
    
    def test(self, test_loader):
        """
        Runs test using the training artifacts
        
        Arguments:
        test_loader -- Test set of Tiny ImageNet dataset
        """
        
        state = CheckpointState.load_checkpoint("./artifacts/checkpoint")

        module = Module(
            "./artifacts/training_model.onnx",
            state,
            "./artifacts/eval_model.onnx",
            device="cpu"
        )
    
        module.eval()  # Set the module in evaluation mode
        losses = []
        metric = evaluate.load('accuracy')

        for _, (data, target) in enumerate(test_loader):
            data = data.to(torch.float32)
            target_logits = self.generate_target_logits(target.tolist())
            
            test_loss, logits = module(data.numpy(), target_logits.numpy())
            print(f"LOSS {test_loss}")
            metric.add_batch(references=target, predictions=self.softmax_activation(logits))
            losses.append(test_loss)

        metrics = metric.compute()
        mean_loss = sum(losses) / len(losses)
        accuracy = metrics["accuracy"]
        print(f'Test Loss: {mean_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        self.save_metrics(mean_loss, accuracy)
        
    def save_metrics(self, loss, accuracy):
        """
        Save the metrics in a json file
        
        Arguments:
        loss -- Current loss of the model
        accuracy -- Current accuracy of the model
        """
        
        self.metrics["losses"].append(loss)
        self.metrics["accuracies"].append(accuracy)
        
        with open("metrics.json", "w") as f:
            json.dump(self.metrics, f)
        
    def __call__(self):
        test_loader = self.load_test_images()
        print("Testing with the training artifacts")
        self.test(test_loader)
        
if __name__ == "__main__":
    test = Test()
    test()
