# Vision Board Box Extractor API

A FastAPI-based web service that automatically extracts individual text boxes from vision board images using computer vision techniques. The API processes images encoded in base64 format and returns extracted boxes as separate base64-encoded images.

## Overview

This service analyzes vision board images to identify and extract distinct content areas (boxes) containing text or other visual elements. It uses OpenCV for image processing, applying binary thresholding and contour detection to locate rectangular regions of interest.

## Key Features

- **Base64 Image Processing**: Accepts images as base64 strings for easy web integration
- **Automatic Box Detection**: Uses OpenCV contour detection to identify rectangular regions
- **Smart Filtering**: Filters out small areas (minimum 1000 pixels) to focus on meaningful content
- **Sorted Output**: Returns boxes ordered from top to bottom based on their position in the image
- **In-Memory Processing**: Processes images entirely in memory without file storage
- **CORS Enabled**: Ready for web application integration

## API Endpoints

- `POST /extract-base64`: Extract boxes from a base64-encoded vision board image
- `GET /`: API information and available endpoints
- `GET /health`: Health check endpoint

## Technology Stack

- **FastAPI**: Modern Python web framework for building APIs
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for array operations
- **Pillow (PIL)**: Python Imaging Library for image format handling
- **Pydantic**: Data validation and serialization
