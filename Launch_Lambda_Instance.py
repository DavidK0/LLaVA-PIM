# This script will attempt to launch the first available Lambda Labs instance.
# It requires a Lambda Labs API key
# If multiple instance become availble simultaneously, the most expensive one will be launched
# If that instance becomes availble in multiple regions simultaneously, the first one is chosen.
#   I do not know how regions are ordered.

import requests
import argparse
import sys
import os
import time
import datetime

# Parse arguments
parser = argparse.ArgumentParser("This script check's Lambda Labs for open capacity.")
parser.add_argument('API_key', help="Your Lambda Labs API key")
args = parser.parse_args()

# Load the API key from a file.
if not os.path.isfile(args.API_key) or not args.API_key.endswith('.txt'):
    print("The API key must be a valid .txt file")
    sys.exit()
with open(args.API_key, 'r') as file:
    API_KEY = file.readline().strip() # just read the first line

# Constants
API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"
SLEEP_TIME = 60

# Headers for authorization and content type
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def check_instance_availability():
    """
    Check the availability of instance types across regions.

    This function fetches and displays the availability and details of various instance types
    from an API. It prints out the name, cost, description, specifications, and availability regions
    of each instance type.

    Returns:
    - list: A list of available instance
            An empty list if there's an error in fetching the data (response status code not 200).

    Prints:
    - A formatted string with details of each available instance type if the fetch is successful.
    - An error message with status code and response text if the fetch fails.
    """

    # Get instance data via a GET request
    response = requests.get(API_BASE_URL + "/instance-types", headers=headers)
    
    def extract_instance_info(instance_data):
        # Extract basic information
        instance_details = instance_data["instance_type"]
        availability = instance_data["regions_with_capacity_available"]

        # Extract details about the instance
        name = instance_details["name"]
        specs = instance_details["specs"]
        price_cents_per_hour = instance_details["price_cents_per_hour"]
        description = instance_details["description"]

        return {"name": name,
                "price": price_cents_per_hour,
                "description": description,
                "specs": specs,
                "availability": availability}

    if response.status_code == 200:
        data = response.json().get('data', {})
        return_list = []
        for instance_name, instance_data in data.items():
            instance_info = extract_instance_info(instance_data)
            if len(instance_info["availability"]) > 0:
                return_list.append(instance_info)
        return return_list

    else:
        # Handle error or logging if necessary
        print(f"Error {response.status_code}: {response.text}")
        return []

def launch_instance(instance_type_name, region_name, ssh_key_name, instance_name):
    """
    Launch a new instance with the specified parameters.
    
    Parameters:
    - instance_type_name (str): The type of the instance to be launched.
    - region_name (str): The name of the region where the instance should be launched.
    - ssh_key_name (str): The name of the SSH key to be associated with the instance for secure access.
    - instance_name (str): The name to be assigned to the new instance.
    
    Returns:
    - dict: The response from the server if the instance is successfully launched (status code 200).
    - None: If there is any error (response status code other than 200), None is returned.

    Prints:
    - A success message with instance launch confirmation.
    - An error message with status code and response text if the launch fails.
    """

    # Payload with details to launch instance
    payload = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": [ssh_key_name],
        "file_system_names": [],
        "quantity": 1,
        "name": instance_name
    }

    # Attempt to launch the instance via a POST request
    response = requests.post(API_BASE_URL + "/instance-operations/launch", json=payload, headers=headers)

    if response.status_code == 200:
        # If the instance is launched successfully
        print("Instance launched successfully.")
        return response.json()
    else:
        # If there was an error
        print(f"Failed to launch instance. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

# Main loop
while True:
    available_instances = check_instance_availability()
    if available_instances:
        print("Instances available:\n")
        for instance in available_instances:
            # Define a format string and use it to format the output
            output_format = "{} {} cents/hour\nDescription: {}\nSpecs: {}\nAvailability: {}\n"
            print(output_format.format(instance["name"],
                                       instance["price"],
                                       instance["description"],
                                       instance["specs"],
                                       instance["availability"]))

        # Sort instance and take the most expensive one (when option are limited I am less worried about the cost)
        chosen_instance = sorted(available_instances, key=lambda x: x["price"], reverse=True)[0]

        # Try to launch the instance
        #print(chosen_instance)
        instance_type_name = chosen_instance['name']
        region_name = chosen_instance['availability'][0]['name'] # pick the first open region, for simplicity
        print(f"\nAttempting to launch {instance_type_name} in {region_name}")
        launch_instance(instance_type_name, region_name, "david_lambda", "david_vision_lang")

        # Add additional logic can go here, e.g., sending an email notification
        break
    else:
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{formatted_datetime} No instances available yet. Checking again in {SLEEP_TIME} seconds...")
        time.sleep(SLEEP_TIME)