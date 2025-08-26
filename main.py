from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import uuid
from typing import List, Dict, Any, Tuple
import base64
from io import BytesIO
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans
import logging

# Configure logging to reduce noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

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


def upload_file_to_opencv_image(file_content: bytes) -> np.ndarray:
    """Convert uploaded file content to OpenCV image"""
    try:
        # Create BytesIO object from file content
        image_stream = BytesIO(file_content)
        
        # Open image with PIL
        pil_image = Image.open(image_stream)
        
        # Convert to RGB if necessary
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        
        # Convert PIL image to numpy array and then to OpenCV format
        image_array = np.array(pil_image)
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise ValueError(f"Unable to process image file: {str(e)}")

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

def merge_contour_group(contour_group: List) -> Tuple[int, int, int, int]:
    """Create a bounding rectangle that encompasses all contours in a group"""
    if len(contour_group) == 1:
        return cv2.boundingRect(contour_group[0])
    
    # Get bounding rectangles for all contours
    rects = [cv2.boundingRect(contour) for contour in contour_group]
    
    # Find the encompassing rectangle
    min_x = min(rect[0] for rect in rects)
    min_y = min(rect[1] for rect in rects)
    max_x = max(rect[0] + rect[2] for rect in rects)
    max_y = max(rect[1] + rect[3] for rect in rects)
    
    return min_x, min_y, max_x - min_x, max_y - min_y

def create_object_mask(image: np.ndarray) -> np.ndarray:
    """Create a mask that identifies foreground objects"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced edge detection
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Multi-scale edge detection
    edges1 = cv2.Canny(filtered, 30, 80)
    edges2 = cv2.Canny(filtered, 50, 120)
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    # Dilate edges to create solid regions
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    edges_dilated = cv2.dilate(edges_combined, kernel_dilate, iterations=3)
    
    # Close gaps in the edges
    mask_filled = edges_dilated.copy()
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Adaptive thresholding to catch objects with uniform colors
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    adaptive_thresh_inv = cv2.bitwise_not(adaptive_thresh)
    
    # Remove small noise from adaptive threshold
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    adaptive_thresh_clean = cv2.morphologyEx(adaptive_thresh_inv, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    # Combine edge-based and adaptive threshold results
    combined_mask = cv2.bitwise_or(mask_filled, adaptive_thresh_clean)
    
    # Final cleanup
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_final, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_final, iterations=1)
    
    return combined_mask

def group_nearby_contours(contours: List, min_distance: int = 30) -> List[List]:
    """Group nearby contours using DBSCAN clustering"""
    if len(contours) == 0:
        return []
    
    # Extract centroids of all contours
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append([cx, cy])
        else:
            # Fallback to bounding rect center
            x, y, w, h = cv2.boundingRect(contour)
            centroids.append([x + w//2, y + h//2])
    
    centroids = np.array(centroids)
    
    # Use DBSCAN to cluster nearby centroids
    clustering = DBSCAN(eps=min_distance, min_samples=1).fit(centroids)
    labels = clustering.labels_
    
    # Group contours by cluster
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(contours[i])
    
    return list(groups.values())

def filter_corner_fragments(contour_groups: List[List]) -> List[List]:
    """Remove small corner fragments that are likely parts of larger objects"""
    if len(contour_groups) <= 1:
        return contour_groups
    
    # Get bounding rectangles for all groups
    group_rects = []
    for i, group in enumerate(contour_groups):
        x, y, w, h = merge_contour_group(group)
        area = w * h
        group_rects.append((x, y, w, h, area, group, i))
    
    # Sort by area (largest first)
    group_rects.sort(key=lambda item: item[4], reverse=True)
    
    filtered_groups = []
    
    for i, (x, y, w, h, area, group, orig_idx) in enumerate(group_rects):
        is_fragment = False
        
        if area < 8000:
            # Compare with larger objects
            for j in range(i):
                larger_x, larger_y, larger_w, larger_h, larger_area, _, larger_orig_idx = group_rects[j]
                
                if larger_area < area * 2:
                    continue
                
                # Check spatial relationship
                small_center_x = x + w // 2
                small_center_y = y + h // 2
                
                # Generous margin for detecting nearby fragments
                margin = 40
                expanded_left = larger_x - margin
                expanded_right = larger_x + larger_w + margin
                expanded_top = larger_y - margin
                expanded_bottom = larger_y + larger_h + margin
                
                # Check if small object is within expanded area of larger object
                if (expanded_left <= small_center_x <= expanded_right and 
                    expanded_top <= small_center_y <= expanded_bottom):
                    
                    # Check distance to corners
                    corners = [
                        (larger_x, larger_y),
                        (larger_x + larger_w, larger_y),
                        (larger_x, larger_y + larger_h),
                        (larger_x + larger_w, larger_y + larger_h)
                    ]
                    
                    min_corner_distance = min(
                        ((small_center_x - corner_x) ** 2 + (small_center_y - corner_y) ** 2) ** 0.5
                        for corner_x, corner_y in corners
                    )
                    
                    # Check distance to edges
                    edge_distances = [
                        abs(small_center_x - larger_x),
                        abs(small_center_x - (larger_x + larger_w)),
                        abs(small_center_y - larger_y),
                        abs(small_center_y - (larger_y + larger_h))
                    ]
                    
                    min_edge_distance = min(edge_distances)
                    
                    if min_corner_distance < 50 or min_edge_distance < 25:
                        is_fragment = True
                        break
                    
                    # Check for overlap
                    overlap_x = max(0, min(x + w, larger_x + larger_w) - max(x, larger_x))
                    overlap_y = max(0, min(y + h, larger_y + larger_h) - max(y, larger_y))
                    overlap_area = overlap_x * overlap_y
                    overlap_ratio = overlap_area / area
                    
                    if overlap_ratio > 0.3:
                        is_fragment = True
                        break
        
        if not is_fragment:
            filtered_groups.append(group)
    
    return filtered_groups

def is_valid_object(x: int, y: int, w: int, h: int, image_shape: Tuple) -> bool:
    """Determine if a detected region represents a valid object to extract"""
    area = w * h
    image_area = image_shape[0] * image_shape[1]
    
    # Size filters
    min_area = 300
    max_area = image_area * 0.7
    
    if area < min_area or area > max_area:
        return False
    
    # Minimum dimensions
    min_width, min_height = 25, 25
    if w < min_width or h < min_height:
        return False
    
    # Aspect ratio
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > 15:
        return False
    
    # Border check
    border_margin = 5
    if (x < border_margin or y < border_margin or 
        x + w > image_shape[1] - border_margin or 
        y + h > image_shape[0] - border_margin):
        return False
    
    return True

def extract_boxes_from_vision_board_base64(image_base64: str) -> List[str]:
    """
    Extract text boxes from a base64 encoded vision board image and return as base64 strings.
    
    Args:
        image_base64: Base64 encoded image string
    
    Returns:
        List of extracted boxes as base64 encoded strings
    """
    image = base64_to_opencv_image(image_base64)
    
    return extract_objects_from_image(image)

def extract_objects_from_image(image: np.ndarray) -> List[str]:
    """Extract meaningful objects from an image and return as base64 strings"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask to identify foreground objects
    mask = create_object_mask(image)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours
    min_contour_area = 200
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    # Group nearby contours
    contour_groups = group_nearby_contours(filtered_contours, min_distance=30)
    
    # Filter out corner fragments
    contour_groups = filter_corner_fragments(contour_groups)
    
    # Create bounding boxes for each group
    object_regions = []
    for group in contour_groups:
        x, y, w, h = merge_contour_group(group)
        
        if is_valid_object(x, y, w, h, image_rgb.shape[:2]):
            object_regions.append((x, y, w, h))
    
    # Sort by vertical position (top to bottom, then left to right)
    object_regions = sorted(object_regions, key=lambda region: (region[1], region[0]))
    
    # Extract each object region
    object_base64_list = []
    
    for i, (x, y, w, h) in enumerate(object_regions):
        # Add padding around the detected object
        padding = 5
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(image_rgb.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(image_rgb.shape[0] - y_padded, h + 2 * padding)
        
        # Extract the object
        obj = image_rgb[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        
        # Convert to base64
        pil_image = Image.fromarray(obj)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        object_base64_list.append(f"data:image/jpeg;base64,{img_base64}")
    
    return object_base64_list

@app.post("/extract-boxes")
async def extract_boxes_file(request: Request) -> Dict[str, Any]:
    """
    Extract text boxes from an uploaded vision board image file.
    
    Args:
        request: FastAPI request object containing form data with 'image' file
        
    Returns:
        Dictionary containing session_id, number of boxes found, and box information with base64 images
    """
    try:
        # Parse multipart form data manually to avoid FastAPI validation issues
        form_data = await request.form()
        
        if "image" not in form_data:
            raise HTTPException(status_code=400, detail="No image file provided in form data")
        
        image_file = form_data["image"]
        
        if not hasattr(image_file, 'read'):
            raise HTTPException(status_code=400, detail="Invalid image file format")
        
        # Read file content
        file_content = await image_file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the image
        opencv_image = upload_file_to_opencv_image(file_content)
        box_base64_list = extract_objects_from_image(opencv_image)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Format results
        results = []
        for i, box_base64 in enumerate(box_base64_list):
            results.append({
                "id": i + 1,
                "image": box_base64
            })
        
        return {
            "session_id": session_id,
            "filename": getattr(image_file, 'filename', 'uploaded_image'),
            "num_boxes_found": len(box_base64_list),
            "boxes": results
        }
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

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
            "POST /extract-boxes": "Extract text boxes from uploaded image file (form-data with 'image' key)",
            "POST /extract-base64": "Extract text boxes from base64 encoded vision board image (JSON body)"
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