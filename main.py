from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import uuid
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="Vision Board Box Extractor V3", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image_base64: str
    filename: str = "image.jpg"

def base64_to_opencv_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(BytesIO(image_data))
        
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def extract_boxes_from_vision_board_base64(image_base64: str) -> List[str]:
    """
    Extract text boxes from a base64 encoded vision board image and return as base64 strings.
    
    Args:
        image_base64: Base64 encoded image string
    
    Returns:
        List of extracted boxes as base64 encoded strings
    """
    image = base64_to_opencv_image(image_base64)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 1000
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])
    
    box_base64_list = []
    
    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        box = image_rgb[y:y+h, x:x+w]
        
        # Convert box to base64
        pil_image = Image.fromarray(box)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        box_base64_list.append(f"data:image/jpeg;base64,{img_base64}")
        
    return box_base64_list

@app.post("/extract-base64")
async def extract_images_base64(image_data: ImageData) -> Dict[str, Any]:
    """
    Extract text boxes from a base64 encoded vision board image.
    
    Args:
        image_data: Object containing base64 encoded image and filename
        
    Returns:
        Dictionary containing session_id, number of boxes found, and box information with base64 images
    """
    if not image_data.image_base64:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    session_id = str(uuid.uuid4())
    
    try:
        box_base64_list = extract_boxes_from_vision_board_base64(image_data.image_base64)
        
        results = []
        for i, box_base64 in enumerate(box_base64_list):
            results.append({
                "id": i + 1,
                "image": box_base64
            })
        
        return {
            "session_id": session_id,
            "num_boxes_found": len(box_base64_list),
            "boxes": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vision Board Box Extractor API V3 - In-Memory Processing",
        "version": "3.0.0",
        "endpoints": {
            "POST /extract-base64": "Extract text boxes from base64 encoded vision board image (returns base64 images)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Add this at the bottom of your existing code, replacing the current if __name__ == "__main__" block:

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 3001))
    uvicorn.run(app, host="0.0.0.0", port=port)