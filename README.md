# SIMULATION OF FEDERATED LEARNING

This project simulates a federated learning scenario on one single computer. The model used here is the Vision Transformer of google, googlevit patch16 224.

In a Federated Learning scenario, users are used to make a critical analysis of the ouptut of a model and retrain it if the ouptput doesn't seem correct to the user.

In order to do this, the users are simulated by chatgpt which provides for each image used the true label. To create this dataset, we use the OPENAI API to loop over all the images of the dataset. At each comparison, if the model prediction is not the same as the prediction of chatgpt-4o, the model is trained on the prediction of chatgpt-4o. 

The provided version runs for `200 users`. Each user has `14 images` to treat. For the training loop, for each image that needs to be trained, data augmentation is performed to provide a total of `7 images` to train on for a given label.

The training occurs in the frontend in javascriptusing the `ONNXRUNTIME JAVASCRIPT API` that handles training. The updated weights are stored in a python backend that updates the model once all the new weights have been received.

## INSTALL REQUIREMENTS   

- Create a new virtual environnement : `python3 -m venv env` and activate it `source env/bin/activate`. Install the requirements for this project `pip install -r requirements.txt`.

## RUN THE PROJECT

- To run the whole project, run `python watcher.py`

## DETAILS OF THE FILES 

### Prepare Dataset for Federated Learning scenario

- `artifacts.py` : Script to create the training artifacts. See `README.md` in the artifacts folder
- `LabelGenerator.py` : Script to generate a dataset of for image classification using chatgpt-4o
- `DatasetCleaner.py` : Script to clean the dataset created by the LabelGenerator.py
- `Base64aaGenerator.py` : Script to generate the string representation of the images using base64 encoding
- `Pipeline.py` : Script to run the whole process of creating the dataset at once
- `prepare_federated.py` : Creates a dataset as a json file for each user
- `Evaluate.py` : Script to run the evaluation process and compare the base model with the model resulting of the federated learning


### Web Server

- `main.py` : Script to launch the python server
- `routes.py` : Script to specify the routes of the python server
- `view.py` : Script to define which function to run to handle a request on a route
- `watcher.py` : Script to restart the server when a file is modified within the project


