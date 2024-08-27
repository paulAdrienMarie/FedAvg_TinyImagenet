import argparse
from FederatedPreparer import FederatedPreparer
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Federated:
    """
    Prepare the Federated Learning and launch it in headless mode on Firefox
    Server has to be launched previously
    
    Attributs:
    nb_users -- Number of users for the Federated Learning
    batch_size -- Size of the subset to attribute to one client 
    communication_round -- Number of model updates
    federated_preparer -- Instance of the FederatedPreparer class 
    url -- URL of the web application
    """
    
    def __init__(self,args):
        """
        Initializes a new instance of the Federated class
        
        Arguments:
        args -- arguments passed at the execution of the script
        """
        
        self.nb_users = args.nb_users
        self.batch_size = args.batch_size
        self.communication_round = args.nb_roc
        self.federated_preparer = FederatedPreparer(self.nb_users,self.batch_size)
        self.url = 'http://localhost:8080'
        
        
    def launch_federated_headless(self, url):
        """
        Connects to the web web page and launches the simulation by clicking 
        on the launch federated button
        
        Arguments:
        url -- URL of the web application
        """

        options = Options()
        options.add_argument('--headless')  # Run in headless mode for no UI
        service = Service('/usr/local/bin/geckodriver')  # Update this path to your WebDriver
        driver = webdriver.Firefox(service=service, options=options)

        try:
            # Open the web application
            driver.get(url)

            # Wait until the page is loaded and a specific element is present (modify the selector as needed)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))  # You can change this to a more specific element
            )

            # Example of filling out input fields
            user_field = driver.find_element(By.ID, "nb_users")
            user_field.clear()  # Clear any existing text
            user_field.send_keys(str(self.nb_users))  # Enter number of users
            
            communication_round_field = driver.find_element(By.ID, "nb_roc")
            communication_round_field.clear()
            communication_round_field.send_keys(str(self.communication_round))

            # Optionally, hit enter after filling out each input (if needed)
            # user_field.send_keys(Keys.RETURN)

            # Click the launch button
            button = driver.find_element(By.ID, 'launch_federated')
            if button:
                print("Button found")
            
            button.click()

            # Wait for the completion of training
            WebDriverWait(driver, 600000000000).until(
                EC.presence_of_element_located((By.ID, 'completion_element_id'))  # Modify to the appropriate element ID
            )

            print("Training Completed")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            # Clean up and close the browser
            driver.quit()
            
        
    def __call__(self):
        """Prepare and launch the Federated Learning"""
        self.federated_preparer()
        self.launch_federated_headless(self.url)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Launch Federated"
    )
    
    parser.add_argument("--nb_users", type=int, help="Number of users")
    parser.add_argument("--batch_size", type=int, help="Size of user batch of images")
    parser.add_argument("--nb_roc", type=int, help="Number of round of communication")
    
    args = parser.parse_args()
    
    obj = Federated(args)
    obj()