import platform
import argparse
import torch
from Artifacts import Artifacts
import os
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import evaluate

assert list(platform.python_version_tuple())[:-1] == ["3", "9"]

class FineTuner:
    """
    Base class for handling common functionality for training and testing.
    
    Attributes:
    path_to_training -- Relative path to the training model
    path_to_eval -- Relative path to the eval model
    path_to_optimizer -- Relative path to the optimizer model
    path_to_checkpoint -- Relative path to the checkpoint file
    path_to_model -- Relative path to the model
    """

    def __init__(self, path_to_training, path_to_eval, path_to_optimizer, path_to_checkpoint, path_to_model):
        self.path_to_training = path_to_training
        self.path_to_eval = path_to_eval
        self.path_to_optimizer = path_to_optimizer
        self.path_to_checkpoint = path_to_checkpoint
        self.path_to_model = path_to_model
        self.model = None
    
    def collate_fn(self, batch):
        """Collates the images and the labels from the TinyImagenet dataset
        
        Returns:
        images -- The images of the dataset
        labels -- The labels of the dataset
        """
        
        images = []
        labels = []
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        for item in batch:
            image = transform(item['image'])
            label = item['label']
            images.append(image)
            labels.append(label)
        
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    def generate_target_logits(self, batch_target):
        """
        Generates the target logits using one hot encoding for training
        
        Arguments:
        batch_target -- Batch of targets containing correct id labels
        """
        
        batch_size = len(batch_target)
        num_classes = 200  # Assuming 200 classes for tiny-imagenet
        target_logits = torch.zeros([batch_size, num_classes], dtype=torch.float32)

        for i, target in enumerate(batch_target):
            target_logits[i, target] = 1

        return target_logits

class Train(FineTuner):
    """
    Class that handles fine-tuning of the google vit model on tinyimagenet dataset
    """

    def load_train_images(self):
        """Loads and returns as a DataLoader the dataset using the collate function"""
        batch_size = 64
        tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')
        train_loader = DataLoader(tiny_imagenet, batch_size=batch_size, collate_fn=self.collate_fn)
        return train_loader
        
    def load_training_modules(self):
        """Loads the necessary training modules to perform training phase"""

        state = CheckpointState.load_checkpoint(self.path_to_checkpoint)
        module = Module(
            self.path_to_training,
            state,
            self.path_to_eval,
            device="cpu"
        )
        optimizer = Optimizer(self.path_to_optimizer, module)
        
        return module, optimizer, state
    
    def train(self, train_loader, epoch):
        """Runs the training loop"""
        module, optimizer, state = self.load_training_modules()
        
        module.train()
        losses = []
        for _, (data, target) in enumerate(train_loader):
           
            # Make sure data is a tensor and properly shaped
            data = data.to(torch.float32)  # Convert to the expected type
            target_logits = self.generate_target_logits(target.tolist())  # Generate batch target logits
            
            train_loss = module(data.numpy(),target_logits.numpy())
            
            print(f"LOSS : {train_loss}")
            optimizer.step()
            module.lazy_reset_grad()
            losses.append(train_loss)

        print(f'Epoch: {epoch+1} - Train Loss: {sum(losses)/len(losses):.4f}')
        
        CheckpointState.save_checkpoint(state, self.path_to_checkpoint)
        module.export_model_for_inferencing(self.path_to_model, ["output"])

    def __call__(self, epoch):
        train_loader = self.load_train_images()
        self.train(train_loader, epoch)

class Test(FineTuner):
    """
    Class to test the resulting fine-tuned model
    """
    
    def load_test_images(self):
        batch_size = 1000
        tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='valid')  # Use the same dataset
        test_loader = DataLoader(tiny_imagenet, batch_size=batch_size, collate_fn=self.collate_fn)
        return test_loader
    
    def softmax_activation(self, logits):
        return np.argmax(logits, axis=-1)
    
    def test(self, test_loader, epoch):
        state = CheckpointState.load_checkpoint("./artifacts/checkpoint")
        module = Module(
            "./artifacts/training_model.onnx",
            state,
            "./artifacts/eval_model.onnx",
            device="cpu"
        )
    
        module.eval()
        losses = []
        
        metric = evaluate.load('accuracy')

        for _, (data, target) in enumerate(test_loader):
            
            data = data.to(torch.float32)
            target_logits = self.generate_target_logits(target.tolist())
            
            test_loss, logits = module(data.numpy(), target_logits.numpy())
            
            metric.add_batch(references=target,predictions=self.softmax_activation(logits=logits))
            losses.append(test_loss)

        metrics = metric.compute()
        print(f'Epoch: {epoch+1} - Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')
        
    def __call__(self, epoch):
        test_loader = self.load_test_images()
        self.test(test_loader, epoch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Create artifacts"
    )
    
    parser.add_argument("--nb_classes", type=int, help="Number of classes of the model")
    
    args = parser.parse_args()
    
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path = os.path.join(os.path.dirname(__file__), "onnx/inference.onnx")
    
    # Create the model and the training artifacts
    obj = Artifacts(
        model_path = model_path,
        artifacts_path = artifacts_path,
        nb_classes = args.nb_classes
    )
        
    train = Train(
        path_to_training = os.path.join(os.path.dirname(__file__), "artifacts","training_model.onnx"),
        path_to_eval = os.path.join(os.path.dirname(__file__), "artifacts","eval_model.onnx"),
        path_to_optimizer = os.path.join(os.path.dirname(__file__), "artifacts","optimizer_model.onnx"),
        path_to_checkpoint = os.path.join(os.path.dirname(__file__), "artifacts","checkpoint"),
        path_to_model=model_path
    )
    test = Test(
        path_to_training = os.path.join(os.path.dirname(__file__), "artifacts","training_model.onnx"),
        path_to_eval = os.path.join(os.path.dirname(__file__), "artifacts","eval_model.onnx"),
        path_to_optimizer = os.path.join(os.path.dirname(__file__), "artifacts","optimizer_model.onnx"),
        path_to_checkpoint = os.path.join(os.path.dirname(__file__), "artifacts","checkpoint"),
        path_to_model=model_path
    )
    
    obj()
    
    for i in range(5):
        train(i)
        test(i)
