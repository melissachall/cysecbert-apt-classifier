#!/usr/bin/env python3
"""
APT Classification REST API - Version Corrigée
Production-ready API for APT group classification
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import json
from datetime import datetime
import logging
import time
import uvicorn

# Import your classifier
from apt_classification_interface import APTClassifier, ClassificationResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="APT Classification API",
    description="Advanced Persistent Threat Group Classification API using CySecBERTMaxPerformance",
    version="1.0.0"
)

# Root endpoint for Docker health and UI
@app.get("/")
async def root():
    return {"status": "ok", "message": "APT Classification API is running."}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Request/Response models
class ClassificationRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.5

class BatchClassificationRequest(BaseModel):
    texts: List[str]
    confidence_threshold: Optional[float] = 0.5

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    top5_probabilities: Dict[str, float]  # CORRECTION: utiliser top5_probabilities
    processing_time: float
    extracted_features: Dict[str, List[str]]
    attribution_factors: List[str]
    timestamp: str

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    summary: Dict

# Global classifier instance
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup"""
    global classifier
    try:
        classifier = APTClassifier()
        logger.info("APT Classifier loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        raise

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification (implement your own logic)"""
    # For demo purposes - in production, implement proper JWT validation
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": classifier is not None
    }

@app.post("/api/v1/classify", response_model=ClassificationResponse)
async def classify_single(
    request: ClassificationRequest,
    token: str = Depends(verify_token)
):
    """Classify a single text input"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    try:
        result = classifier.classify(request.text, request.confidence_threshold)
        
        return ClassificationResponse(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            top5_probabilities=result.top5_probabilities,  # CORRECTION
            processing_time=result.processing_time,
            extracted_features=result.extracted_features,
            attribution_factors=result.attribution_factors,
            timestamp=result.timestamp
        )
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    request: BatchClassificationRequest,
    token: str = Depends(verify_token)
):
    """Classify multiple texts in batch"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    try:
        start_time = time.time()
        results = []
        
        # CORRECTION: classifier n'a pas de méthode classify_batch
        for text in request.texts:
            result = classifier.classify(text, request.confidence_threshold)
            results.append(result)
        
        total_time = time.time() - start_time
        
        response_results = [
            ClassificationResponse(
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                top5_probabilities=result.top5_probabilities,  # CORRECTION
                processing_time=result.processing_time,
                extracted_features=result.extracted_features,
                attribution_factors=result.attribution_factors,
                timestamp=result.timestamp
            )
            for result in results
        ]
        
        # Calculate summary statistics
        confidences = [r.confidence for r in results]
        predictions = [r.predicted_class for r in results]
        
        summary = {
            "total_processed": len(results),
            "total_time": total_time,
            "average_confidence": sum(confidences) / len(confidences),
            "average_processing_time": sum(r.processing_time for r in results) / len(results),
            "prediction_distribution": {pred: predictions.count(pred) for pred in set(predictions)}
        }
        
        return BatchClassificationResponse(
            results=response_results,
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/classify/file")
async def classify_file(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    token: str = Depends(verify_token)
):
    """Classify text from uploaded file"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        result = classifier.classify(text, confidence_threshold)
        
        return ClassificationResponse(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            top5_probabilities=result.top5_probabilities,  # CORRECTION
            processing_time=result.processing_time,
            extracted_features=result.extracted_features,
            attribution_factors=result.attribution_factors,
            timestamp=result.timestamp
        )
    
    except Exception as e:
        logger.error(f"File classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/classes")
async def get_classes(token: str = Depends(verify_token)):
    """Get available APT classes"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    return {
        "classes": classifier.class_names,
        "total_classes": len(classifier.class_names),
        "model_info": {
            "architecture": "CySecBERTMaxPerformance",
            "device": str(classifier.device)
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "apt_classification_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )