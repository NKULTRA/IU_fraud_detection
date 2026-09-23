# From Model to Production - Fraud Detection

This project is for the course assignment "From Model to Production" (DLBDSMT01). It demonstrates how to build, train, validate, and deploy a fraud detection model in a realistic ML workflow.

## Overview

The repository includes:
- data ingestion and preprocessing
- model training and MLflow experiment tracking
- drift detection and retraining checks
- model promotion to a production alias
- a Flask API for serving predictions
- automated tests for the API

## Project structure

- `src/` – core Python modules for data handling, preprocessing, training, and API
- `scripts/` – operational scripts such as drift simulation, model promotion, and blob sync
- `tests/` – validation tests
- `data/` – input and generated datasets
- `config/` – YAML configuration files
- `mlruns/` – MLflow tracking data

## Typical workflow

1. Prepare data and configuration
2. Train the model with MLflow tracking
3. Evaluate drift and decide whether retraining is needed
4. Promote the selected model version to `production`
5. Use the Flask API to score new transactions

## Local setup

Install dependencies from `requirements.txt` and use the project virtual environment for training and testing.

## Notes

This repository is an example end-to-end ML pipeline for fraud detection and is intended to show the full model lifecycle from experimentation to production deployment.
