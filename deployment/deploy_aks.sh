#!/bin/bash
set -e

# Configuration
RESOURCE_GROUP="DeepfakeDetectionRG"
LOCATION="eastus"
AKS_CLUSTER_NAME="DeepfakeCluster"
ACR_NAME="deepfakeregistry"
NAMESPACE="deepfake-detection"
MODEL_VERSION="v1"
APP_NAME="deepfake-detector"

# Create resource group if it doesn't exist
echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output none || true

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard --output none || true

# Create AKS cluster if it doesn't exist
echo "Creating AKS Cluster..."
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_CLUSTER_NAME \
    --node-count 2 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --node-vm-size Standard_DS3_v2 \
    --output none || echo "AKS cluster already exists"

# Attach ACR to AKS for authentication
echo "Attaching ACR to AKS..."
az aks update -n $AKS_CLUSTER_NAME -g $RESOURCE_GROUP --attach-acr $ACR_NAME --output none || true

# Get AKS credentials
echo "Getting AKS credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME --overwrite-existing

# Create Kubernetes namespace if it doesn't exist
echo "Creating Kubernetes namespace..."
kubectl create namespace $NAMESPACE || true

# Create Azure Storage Account for blob storage
STORAGE_ACCOUNT_NAME="deepfakestorage$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)"
echo "Creating Storage Account: $STORAGE_ACCOUNT_NAME"
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

# Create blob container
echo "Creating blob container..."
az storage container create --name deepfake-videos \
    --connection-string "$STORAGE_CONNECTION_STRING" \
    --output none || true

# Create Content Safety resource
CONTENT_SAFETY_NAME="DeepfakeContentSafety"
echo "Creating Content Safety resource..."
az cognitiveservices account create \
    --name $CONTENT_SAFETY_NAME \
    --resource-group $RESOURCE_GROUP \
    --kind ContentSafety \
    --sku S0 \
    --location $LOCATION \
    --output none || true

# Get Content Safety endpoint and key
CONTENT_SAFETY_ENDPOINT=$(az cognitiveservices account show \
    --name $CONTENT_SAFETY_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.endpoint \
    --output tsv)

CONTENT_SAFETY_KEY=$(az cognitiveservices account keys list \
    --name $CONTENT_SAFETY_NAME \
    --resource-group $RESOURCE_GROUP \
    --query key1 \
    --output tsv)

# Build and push Docker image
echo "Building and pushing Docker image to ACR..."
cd ../api

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)

# Build and push with ACR
az acr build --registry $ACR_NAME --image ${APP_NAME}:${MODEL_VERSION} .

# Create Kubernetes secrets for Azure Storage and Content Safety
echo "Creating Kubernetes secrets..."
kubectl create secret generic azure-storage-secret \
    --namespace $NAMESPACE \
    --from-literal=connection-string="$STORAGE_CONNECTION_STRING" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic content-safety-secret \
    --namespace $NAMESPACE \
    --from-literal=endpoint="$CONTENT_SAFETY_ENDPOINT" \
    --from-literal=key="$CONTENT_SAFETY_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Create Kubernetes deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $APP_NAME
  template:
    metadata:
      labels:
        app: $APP_NAME
    spec:
      containers:
      - name: $APP_NAME
        image: $ACR_LOGIN_SERVER/${APP_NAME}:${MODEL_VERSION}
        ports:
        - containerPort: 5000
        env:
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: connection-string
        - name: AZURE_CONTENT_SAFETY_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: content-safety-secret
              key: endpoint
        - name: AZURE_CONTENT_SAFETY_KEY
          valueFrom:
            secretKeyRef:
              name: content-safety-secret
              key: key
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
EOF

# Create Kubernetes service
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME
  namespace: $NAMESPACE
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5000
  selector:
    app: $APP_NAME
EOF

# Wait for service external IP
echo "Waiting for service external IP..."
external_ip=""
while [ -z $external_ip ]; do
    echo "Waiting for endpoint..."
    external_ip=$(kubectl get svc $APP_NAME --namespace $NAMESPACE --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}")
    [ -z "$external_ip" ] && sleep 10
done

echo "Deepfake detection API is available at: http://$external_ip"
echo "Deployment completed successfully!"