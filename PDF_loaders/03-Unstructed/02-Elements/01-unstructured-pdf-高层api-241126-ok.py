# 参考
from dotenv import load_dotenv
load_dotenv()

import os
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
import json

# Set your Unstructured API key
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY") 
print(f"API Key: {UNSTRUCTURED_API_KEY}")

# Initialize the client
client = UnstructuredClient(
    api_key_auth=UNSTRUCTURED_API_KEY,
    server_url="https://api.unstructuredapp.io/general/v0/general",
)

# Specify the file path
file_path = "01.loading/PDF/山西-en.pdf"

# Read the file content
with open(file_path, "rb") as f:
    file_content = f.read()

# Create the partition request
partition_request = operations.PartitionRequest(
    partition_parameters=shared.PartitionParameters(
        files=shared.Files(
            content=file_content,
            file_name=file_path,
        ),
        strategy="fast",  # Partitioning strategy
        chunking_strategy="by_title",  # Chunking strategy,
        # include_metadata=True,  # Include metadata in the response
    ),
)

# Send the partition request to the API
response = client.general.partition(partition_request)

# Check the response
if response.status_code == 200:
    elements = json.loads(response.raw_response.text)
    for element in elements:
        text = element.get("text")
        metadata = element.get("metadata", {})
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")
else:
    print(f"Error: {response.status_code}")

