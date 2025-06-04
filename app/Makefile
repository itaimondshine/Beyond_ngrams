# Makefile for building and pushing Docker images to Google Cloud Run

# Variables
PROJECT_ID := cabby-316408
SERVICE_NAME ?= eval
IMAGE_NAME := gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)
REGION := asia-east1
ENV_VARS := LANGUAGE=$(LANGUAGE),OPEN_AI_KEY=$(OPEN_AI_KEY),N_TASKS=$(N_TASKS),DEFAULT_ARGS=$(DEFAULT_ARGS)


# Default target
all: build push deploy

# Build the Docker image
build:
	docker build -t eval .

up:
	docker run -v .:/app_instructor -p 5002:5002 -e LANGUAGE='spanish' -e OPEN_AI_KEY= eval


# Push the Docker image to Google Container Registry
push:
	docker push $(IMAGE_NAME)

gcloud-build:
	gcloud builds submit --tag $(IMAGE_NAME)

# Deploy the image to Cloud Run
gcloud-deploy:
	gcloud run deploy $(SERVICE_NAME) \
		--image $(IMAGE_NAME) \
		--memory 1024Mi \
		--region $(REGION) \
		--set-env-vars $(ENV_VARS) \
		--platform managed \
		--allow-unauthenticated


# Clean up local Docker images
clean:
	docker rmi $(IMAGE_NAME)

.PHONY: all build push deploy clean
