import { run } from "./federated.js";

const launch_federated = document.getElementById("launch_federated");

launch_federated.addEventListener("click", async function (e) {
  // Retrieve the input field values when the button is clicked
  const nb_users = document.getElementById("nb_users").value;
  const nb_roc = document.getElementById("nb_roc").value;

  // Log the values to the console
  console.log("Number of users:", nb_users);
  console.log("Number of communication rounds:", nb_roc);
  
  // Call the function to run federated learning (uncomment if ready)
  await run(nb_users, nb_roc);
  location.reload(true);  // Optionally reload the page
});
