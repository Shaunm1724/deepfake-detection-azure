#!/bin/bash
set -e

# Configuration
RESOURCE_GROUP="DeepfakeDetectionRG"
LOCATION="eastus"
STORAGE_ACCOUNT_NAME="deepfakefuncstorage$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)"
FUNCTION_APP_NAME="deepfake-functions"
SERVICE_PLAN_NAME="deepfake-service-plan"
AKS_SERVICE_NAME="deepfake-detector"
NAMESPACE="deepfake-detection"

# Create resource group if it doesn't exist
echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output none || true

# Create Storage Account for Azure Functions if it doesn't exist
echo "Creating Storage Account for Azure Functions..."
az storage account create \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none || true

# Get Storage Account connection string
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --output tsv)

# Create Function App Service Plan
echo "Creating Function App Service Plan..."
az functionapp plan create \
    --name $SERVICE_PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku B1 \
    --output none || true

# Create Function App
echo "Creating Function App..."
az functionapp create \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $SERVICE_PLAN_NAME \
    --storage-account $STORAGE_ACCOUNT_NAME \
    --runtime python \
    --runtime-version 3.9 \
    --functions-version 4 \
    --os-type Linux \
    --output none || true

# Get API endpoint from Kubernetes service
API_IP=$(kubectl get svc $AKS_SERVICE_NAME --namespace $NAMESPACE --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}")

if [ -z "$API_IP" ]; then
    echo "ERROR: Could not retrieve API IP address from Kubernetes service"
    exit 1
fi

API_ENDPOINT="http://$API_IP"

# Set Function App settings
echo "Configuring Function App settings..."
az functionapp config appsettings set \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
    DEEPFAKE_API_ENDPOINT="$API_ENDPOINT" \
    AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONNECTION_STRING" \
    FUNCTIONS_EXTENSION_VERSION="~4" \
    FUNCTIONS_WORKER_RUNTIME="python" \
    --output none

# Create function project locally
echo "Creating function project locally..."
mkdir -p ./function_app
cd ./function_app

# Initialize function app
func init --worker-runtime python --docker

# Create HTTP trigger function
func new --template "HTTP trigger" --name ProcessVideo

# Update function code
cat > ./ProcessVideo/__init__.py << 'EOL'
import logging
import azure.functions as func
import requests
import os
import json
import tempfile
from azure.storage.blob import BlobServiceClient, ContentSettings

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get API endpoint from settings
    api_endpoint = os.environ.get('DEEPFAKE_API_ENDPOINT')
    if not api_endpoint:
        return func.HttpResponse(
            "DEEPFAKE_API_ENDPOINT configuration is missing.",
            status_code=500
        )
    
    # Check if the request contains a file
    if not req.files:
        return func.HttpResponse(
            "Please upload a video file in the request.",
            status_code=400
        )
    
    # Get the file from the request
    video_file = req.files.get('video')
    if not video_file:
        return func.HttpResponse(
            "No video file found in the request.",
            status_code=400
        )
    
    try:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
            temp.write(video_file.read())
            temp_path = temp.name
        
        # Send the video to the deepfake detection API
        with open(temp_path, 'rb') as f:
            files = {'video': (video_file.filename, f)}
            response = requests.post(f"{api_endpoint}/predict", files=files)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if response.status_code != 200:
            return func.HttpResponse(
                f"Error from deepfake detection API: {response.text}",
                status_code=response.status_code
            )
        
        # Return the detection results
        return func.HttpResponse(
            response.text,
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )
EOL

# Update function.json
cat > ./ProcessVideo/function.json << 'EOL'
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "function",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": [
        "post"
      ]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
EOL

# Update requirements.txt
cat > ./requirements.txt << 'EOL'
azure-functions
requests
azure-storage-blob
EOL

# Deploy the function app
echo "Deploying function app..."
func azure functionapp publish $FUNCTION_APP_NAME --python

echo "Azure Functions deployment completed successfully!"
echo "Function endpoint: https://$FUNCTION_APP_NAME.azurewebsites.net/api/ProcessVideo"