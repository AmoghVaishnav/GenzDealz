import csv
import json

# Input and output file paths
csv_file_path = 'F:\GenzDealz\FInalDataset.csv'  # replace with your CSV file path
json_file_path = 'output_dataset.json'

# Function to convert CSV to JSON
def csv_to_json(csv_file_path, json_file_path):
    json_data = []

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            # Create a new JSON object for each row, excluding null or empty values
            json_object = {key: value for key, value in row.items() if value}

            # Convert specific fields to integer
            if 'Customer Age' in json_object:
                json_object['Customer Age'] = int(json_object['Customer Age'])
            if 'Quantity' in json_object:
                json_object['Quantity'] = int(json_object['Quantity'])
            if 'Total Purchase Amount' in json_object:
                json_object['Total Purchase Amount'] = int(json_object['Total Purchase Amount'])
            if 'Product Price' in json_object:
                json_object['Product Price'] = int(json_object['Product Price'])

            json_data.append(json_object)

    # Save the JSON data to a file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

# Convert the CSV to JSON
csv_to_json(csv_file_path, json_file_path)
