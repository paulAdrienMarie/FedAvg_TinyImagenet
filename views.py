from aiohttp import web
import os
from ModelUpdater import ModelUpdater
from Artifacts import Artifacts
from Evaluate import Test

# Initialize paths to artifacts and models
model_path = os.path.join(os.path.dirname(__file__), "onnx/inference.onnx")
artifacts_path = os.path.join(os.path.dirname(__file__),"artifacts")

# Initialize numerical constant
nb_classes = 200

# Initialize Test and Artifacts instances
test = Test()
artifacts = Artifacts(
            artifacts_path=artifacts_path,
            model_path=model_path,
            nb_classes=nb_classes
        )

# Global variable to store the number of users
model_updater = None
nb_users = None
nb_roc = None

async def set_num_user(request):
    """Saves the parameters given at execution in the backend"""
    global nb_users  # Declare the global variable
    global model_updater
    global nb_roc
    
    try:
        
        data = await request.json()  # Await the JSON data
        nb_users = int(data.get("nb_users"))  # Set the global variable
        nb_roc = int(data.get("nb_roc"))
        
        model_updater = ModelUpdater(
            model_path=model_path,
            nb_users=nb_users,
            nb_roc=nb_roc
        )
        
        return web.json_response({"message": f"Number of users set to {nb_users}"})
    
    except Exception as e:
        
        error_message = str(e)
        response_data = {"error": error_message}
        return web.json_response(response_data, status=500)

async def update_model(request):
    """Handles storage and update of model's parameters"""
    global nb_users  # Access the global variable
    
    try:
        
        data = await request.json()
        updated_weights = data.get("updated_weights")
        values = updated_weights["cpuData"].values()
        list_values = list(values)
        user_id = data.get("user_id")
        epoch = data.get("epoch")
        
        print(f'Received data of user {user_id} - Epoch {epoch+1}/{nb_roc}')
        model_updater.update_weights(updated_weights=list_values)
        
        # Ensure nb_users is set before using it
        if nb_users is None:
            return web.json_response({"error": "Number of users not set"}, status=400)
        
        if len(list(model_updater.parameters.items())) % nb_users == 0:
            print("Updating the model parameters")
            response_data = model_updater.update_model()
            print("Generating the new training artifacts based on the new model")
            artifacts.gen_artifacts()
            print("Start testing phase")
            test()
        else:
            response_data = {"message": "User data added, waiting for more data to update the model"}
        
        return web.json_response(response_data)
    
    except Exception as e:
        error_message = str(e)
        response_data = {"error": error_message}
        return web.json_response(response_data, status=500)

async def index(request):
    """Serves index.html file"""
    print("New connection")
    return web.FileResponse("./index.html")

async def style(request):
    """Serves style.css file"""
    return web.FileResponse("./style.css")