import * as ort from "/dist/ort.training.wasm.min.js";

ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let BATCHSIZE = 5;
let NUMUSERS = null;
let NUMEPOCHS = null;

/**
 * Runs the federated learning scenario for a 100 users
 * @async
 * @runFederated
 * @returns {Promise<Void>}
 */
async function runFederated(com_round) {
  
  let completedUsers = 0;

  /**
   * Stop the current thread for a given time
   * @sleep
   * @param {Number} ms - Number of ms to wait
   * @returns {Promise<void>}
   */
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Runs the federated learning by batch of 20 users, 20 users run in parallel, in separate workers
   * @async
   * @param {Number} startIndex - Index to start the next batch of users
   * @returns {Promise<Void>}
   */
  async function runBatch(startIndex, com_round) {
    let promises = [];
    let workers = []; // Array to keep track of workers

    for (let j = 0; j < BATCHSIZE; j++) {
      let userIndex = startIndex + j;
      if (userIndex >= NUMUSERS) break;
      console.log(`COMMUNICATION ROUND ${com_round}/${NUMEPOCHS} - Creating Worker for user ${userIndex + 1}`);

      let worker = new Worker("/script/worker.js", {
        type: "module",
      });

      // Send the data to the created Worker
      let data = {
        userId: userIndex + 1,
        epoch: com_round,
        nb_user: NUMUSERS
      };
      worker.postMessage(data);

      // Push the worker to the workers array
      workers.push(worker);

      promises.push(
        new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            console.log(`User ${e.data.userId} completed training.`);
            resolve();
          };

          worker.onerror = (e) => {
            console.error(`Error in worker for user ${userIndex + 1}:`, e);
            reject(e);
          };
        })
      );
    }
    // Wait for all promises to resolve
    await Promise.all(promises);
    console.log(promises)
    console.log("Terminating all workers");
    // Terminate all workers
    workers.forEach((worker) => worker.terminate());

    completedUsers += BATCHSIZE;
  }
  
  for (let i = 0; i < NUMUSERS; i += BATCHSIZE) {
    await runBatch(i, com_round);
  }

  
}

export async function run(nb_users, nb_roc) {

  NUMUSERS = Number(nb_users);
  NUMEPOCHS = Number(nb_roc);
  fetch("/set_nb_users", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      nb_users: nb_users,
      nb_roc: nb_roc
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
    })
    .catch((error) => {
      console.log("Error:", error);
    });

  
  for (let epoch=0; epoch<NUMEPOCHS; epoch++) {
    await runFederated(epoch);
  }

  let completion_element = document.createElement("p");
  completion_element.id = "completion_element_id";
  completion_element.innerText = "Process terminated";
  let federated_button = document.getElementById("launch_federated");
  federated_button.appendChild(completion_element);
}
