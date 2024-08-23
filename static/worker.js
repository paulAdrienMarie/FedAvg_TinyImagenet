import "/dist/tf.min.js";
import * as ort from "/dist/ort.training.wasm.min.js";

// Set up wasm paths
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

// Initialization of both inference training session
let trainingSession = null;
let inferenceSession = null;

// Number of epochs
let NUMEPOCHS = 5;

// Paths to the training artifacts
const ARTIFACTS_PATH = {
  checkpointState: "/artifacts/checkpoint",
  trainModel: "/artifacts/training_model.onnx",
  evalModel: "/artifacts/eval_model.onnx",
  optimizerModel: "/artifacts/optimizer_model.onnx",
};

// Path to the base model
let MODEL_PATH = "/onnx/inference.onnx";

// Worker code for message handling
self.addEventListener("message", async (event) => {
  let data = event.data;
  let userId = data.userId;
  let epoch = data.epoch;
  let nb_users = data.nb_user;
  let index = getRandomNumber(1,nb_users);
  console.log(index)
  var user_file = await loadJson(`/script/dataset/user_${index}.json`);

  console.log(`CURRENTLY RUNNING USER ${userId} - EPOCH ${epoch}/50`);
  console.log(`LOADING TRAINING SESSION FOR USER ${userId}`);

  // Load the Training session of the current user
  await loadTrainingSession(ARTIFACTS_PATH);
  // Loop over the items of the dataset
  for (const id in user_file) {
    let true_label = user_file[id].label;
    let base64 = user_file[id].base64;
    // Get the label predicted by the base model
    let label = await predict(base64);
    console.log(
      `True label is ${true_label}, ONNX model predicted ${label}`
    );
    // Compare the label predicted by the base model to the true label
    if (true_label !== label) {
      await train(base64, true_label); // Retrain the model on the misclassified image
    }
  }
  // Retrieve the updated weights from the training session
  let params = await trainingSession.getContiguousParameters(true);
  console.log(`Making requests for user ${userId}`);

  // Send the updated weights to the backend server for storage
  let start = Date.now();
  fetch("/update_model", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      updated_weights: params,
      user_id: userId,
      epoch: epoch
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      self.postMessage({
        userId: userId,
      });
      let request_time = Date.now() - start;
      console.log(`Request time : ${request_time} milliseconds`);
      console.log("Model parameters updated");
      console.log(`Request done for user ${userId}`);
    })
    .catch((error) => {
      console.log("Error:", error);
    });
});

self.onerror = function (error) {
  console.error("Worker error:", error);
};

/**
 * Get a random number between min and max
 * @getRandomNumber
 * @param {Int} min - min value
 * @param {Int} max - max value
 * @returns {Int}
 */
function getRandomNumber(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Creates the expected logits output for a given class
 * @createTargetTensor
 * @param {String} new_class - The correct class of the image
 * @returns {Tensor} A tensor with a highest value at the index of the correct class
 */
async function createTargetTensor(new_class) {
  const config = await loadJson("/script/config.json");
  const index = config.label2id[new_class];
  const shape = [1, 200];
  const low_value = 0;

  const size = shape.reduce((a, b) => a * b);
  let data = new Float32Array(size).fill(low_value);
  data[index] = 1;

  return new ort.Tensor("float32", data, shape);
}
/**
 * Instantiate an inference session
 * @async
 * @loadInferenceSession
 * @param {String} model_path - Path to the base model
 * @returns {Promise<void>}
 */
async function loadInferenceSession(model_path) {
  console.log("Loading Inference Session");

  try {
    inferenceSession = await ort.InferenceSession.create(model_path);
    console.log("Inference Session successfully loaded");
  } catch (err) {
    console.log("Error loading the Inference Session:", err);
    throw err;
  }
}

/**
 * Instantiate a training session
 * @async
 * @loadTrainingSession
 * @param {Object} training_paths - Paths to the training artifacts
 * @returns {Promise<void>}
 */
async function loadTrainingSession(training_paths) {
  console.log("Trying to load Training Session");

  try {
    trainingSession = await ort.TrainingSession.create(training_paths);
    console.log("Training session loaded");
  } catch (err) {
    console.error("Error loading the training session:", err);
    throw err;
  }
}

/**
 * Loads JSON from a given URL
 * @async
 * @loadJson
 * @param {String} url - URL of the file
 * @returns {Promise<Object>}
 */
async function loadJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading JSON", error);
    return null;
  }
}

/**
 * Converts an image in base64 string format into a tensor of shape [1, 3, 224, 224]
 * @async
 * @toTensor
 * @param {String} base64 - base64 encoded representation of the image
 * @returns {Promise<ort.Tensor>}
 */
async function toTensor(base64) {
  // Fetch the image and create a bitmap from it
  const imgBlob = await fetch(base64).then((res) => res.blob());
  const imgBitmap = await createImageBitmap(imgBlob);

  // Define the desired input size for the model (224x224 for many models like ViT)
  const targetWidth = 224;
  const targetHeight = 224;
  
  // Create an offscreen canvas for resizing the image
  const canvas = new OffscreenCanvas(targetWidth, targetHeight);
  const ctx = canvas.getContext("2d");

  // Resize and draw the image to the canvas
  ctx.drawImage(imgBitmap, 0, 0, targetWidth, targetHeight);

  // Get the image data from the canvas
  const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);

  // Prepare a Float32Array for the tensor data
  const inputSize = targetWidth * targetHeight * 3; // 3 channels (RGB)
  const dataFromImage = new Float32Array(inputSize);

  // Fill the array with normalized RGB values
  for (let i = 0; i < targetWidth * targetHeight; i++) {
    const r = imageData.data[i * 4];     // Red channel, normalize to [0, 1]
    const g = imageData.data[i * 4 + 1]; // Green channel, normalize to [0, 1]
    const b = imageData.data[i * 4 + 2]; // Blue channel, normalize to [0, 1]

    // Write to the data array in the order of channels (C, H, W)
    dataFromImage[i] = r;
    dataFromImage[targetWidth * targetHeight + i] = g;
    dataFromImage[2 * targetWidth * targetHeight + i] = b;
  }

  // Define the shape of the tensor
  const shape = [1, 3, targetHeight, targetWidth];
  
  // Create the tensor
  const imageTensor = new ort.Tensor("float32", dataFromImage, shape);
  return imageTensor;
}

/**
 * Normalizes input image represented as a tensor
 * @async
 * @preprocessImage
 * @param {ort.Tensor} tensor - Image as a Tensor
 * @returns {Promise<ort.Tensor>}
 */
async function preprocessImage(tensor) {
  const conf = await loadJson("/script/preprocessor_config.json");
  const imageMean = conf.image_mean; // Ensure this is an array of length 3, e.g., [0.485, 0.456, 0.406]
  const imageStd = conf.image_std;   // Ensure this is an array of length 3, e.g., [0.229, 0.224, 0.225]

  // Validate the configuration
  if (imageMean.length !== 3 || imageStd.length !== 3) {
      console.error("Configuration error: image_mean and image_std should be arrays of length 3.");
      return null;
  }

  let data = await tensor.getData();

  // Normalize the image data: (value / 255 - mean) / std
  for (let i = 0; i < data.length; i += 3) {
      // Normalize each channel separately
      for (let j = 0; j < 3; j++) {
          data[i + j] = (data[i + j] / 255.0 - imageMean[j]) / (imageStd[j] || 1); // Use 1 to avoid division by zero
      }
  }

  return new ort.Tensor("float32", data, [1, 3, 224, 224]);
}


/**
 * Performs softmax activation on logits in array format
 * @softmax
 * @param {Array[Float32]} logits - Raw outputs of the onnx model
 * @returns {Array[Float32]} Probability distribution in an array
 */
function softmax(logits) {
  return logits.map((value, index) => {
    return (
      Math.exp(value) /
      logits
        .map((y) => Math.exp(y))
        .reduce((a, b) => a + b)
    );
  });
}

/**
 * Sorts an array in descending order
 * @argsort
 * @param {Array[]} array - The array to be sorted
 * @returns {Promise<Array>} The sorted array
 */
function argsort(array) {
  const arrayWithIndices = Array.from(array).map((value, index) => ({
    value,
    index,
  }));

  arrayWithIndices.sort((a, b) => b.value - a.value);

  return arrayWithIndices.map((item) => item.index);
}

/**
 * Given the base64 image, returns the predicted class
 * @async
 * @predict
 * @param {String} base64 - Base64 representation of the image
 * @returns {Promise<Object>} Predicted class with probability score
 */
async function predict(base64) {
  // Check if the inference session has been loaded
  if (!inferenceSession) {
    await loadInferenceSession(MODEL_PATH);
  }

  const imageTensor = await toTensor(base64);
  const preprocessedImage = await preprocessImage(imageTensor);

  const feeds = {
    "pixel_values": preprocessedImage
  };
  const results = await inferenceSession.run(feeds);
  const logits = results.output.cpuData;

  const prob = softmax(logits);
  const top5Classes = argsort(prob).slice(0, 1);
  const config = await loadJson("/script/config.json");
  const label = config.id2label[top5Classes[0]];
  return label;
}

/**
 * Perform one training step with a given image and its correct label
 * @async
 * @train
 * @param {String} base64 - Base64 representation of the image
 * @param {String} true_label - Correct label of the image
 * @returns {Promise<void>}
 */
async function train(base64, true_label) {
  // Check if the training session has been loaded
  if (!trainingSession) {
    await loadTrainingSession(ARTIFACTS_PATH);
  }

  const imageTensor = await toTensor(base64);
  const preprocessedImage = await preprocessImage(imageTensor);

  const startTrainingTime = Date.now();
  console.log("Training started");

  const target_tensor = await createTargetTensor(true_label);

  for (let epoch = 0; epoch < NUMEPOCHS; epoch++) {
    await runTrainingEpoch(preprocessedImage, epoch, target_tensor);
  }

  const trainingTime = Date.now() - startTrainingTime;
  console.log(`Training completed in ${trainingTime} milliseconds`);
}


/**
 * Runs a single epoch of the training loop
 * @runTrainingEpoch
 * @param {Set[Tensor]} images - Set of augmented images of the image to train on
 * @param {Number} epoch - Current epoch
 * @param {Tensor} target_tensor - The target tensor
 */
async function runTrainingEpoch(image, epoch, target_tensor) {
  const epochStartTime = Date.now();
  const lossNodeName = trainingSession.handler.outputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${NUMEPOCHS} | Starting Training ... `
  );

  const feeds = {
    pixel_values: image,
    target : target_tensor,
  };

  const results = await trainingSession.runTrainStep(feeds);
  const loss = results[lossNodeName].data;

  console.log(`LOSS: ${loss}`);

  await trainingSession.runOptimizerStep();
  await trainingSession.lazyResetGrad();

  const res = await trainingSession.runEvalStep(feeds);
  console.log("Run eval step", res);

  const epochTime = Date.now() - epochStartTime;
  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}
