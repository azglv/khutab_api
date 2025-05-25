from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from typing import List, Dict, Optional
import re
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
import shutil
from pathlib import Path
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for any frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 Configuration - Updated with correct values
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "manaratmodel")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "Khutab_model_v2.zip")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "me-south-1")

# Global variables for models
model = None
tokenizer = None
urdu_model = None
urdu_tokenizer = None

def download_model_from_s3():
    """Download zipped model from S3 and extract it to temporary directory"""
    try:
        logger.info(f"Downloading model from S3 bucket: {S3_BUCKET_NAME}")
        
        # Create S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Create temporary directory for model
        temp_dir = tempfile.mkdtemp(prefix="khutab_model_")
        zip_file_path = os.path.join(temp_dir, "model.zip")
        extract_dir = os.path.join(temp_dir, "extracted")
        
        logger.info(f"Created temporary directory: {temp_dir}")
        
        try:
            # Download the zip file
            logger.info(f"Downloading {S3_MODEL_KEY} to {zip_file_path}")
            s3_client.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, zip_file_path)
            
            # Extract the zip file
            logger.info(f"Extracting zip file to {extract_dir}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Remove the zip file to save space
            os.remove(zip_file_path)
            
            # Find the actual model directory (might be nested)
            model_path = None
            for root, dirs, files in os.walk(extract_dir):
                # Look for key model files
                if any(f in files for f in ['config.json', 'pytorch_model.bin', 'model.safetensors']):
                    model_path = root
                    break
            
            if model_path is None:
                logger.error("Could not find model files in extracted archive")
                return None
                
            logger.info(f"Model extracted successfully to {model_path}")
            return model_path
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            return None
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            return None
            
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return None
    except Exception as e:
        logger.error(f"Error downloading model from S3: {e}")
        return None

def load_models():
    """Load models with fallback strategy"""
    global model, tokenizer, urdu_model, urdu_tokenizer
    
    # First try to download and load custom model from S3
    model_dir = download_model_from_s3()
    
    if model_dir and os.path.exists(model_dir):
        try:
            logger.info(f"Loading custom tokenizer from {model_dir}")
            tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
            logger.info(f"Loading custom model from {model_dir}")
            model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
            logger.info("Custom model loaded successfully from S3")
        except Exception as e:
            logger.error(f"Error loading custom model from S3: {e}")
            model = None
            tokenizer = None
    
    # Fallback to standard model if custom model loading fails
    if model is None or tokenizer is None:
        try:
            logger.info("Loading fallback standard model")
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            logger.info("Standard model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading standard model: {e}")
            raise e
    
    # Load base M2M100 model for Urdu translation
    try:
        logger.info("Loading Urdu model")
        urdu_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        urdu_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        logger.info("Urdu model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Urdu model: {e}")
        raise e
    
    # Clean up temporary directory if it exists
    if model_dir and os.path.exists(model_dir):
        try:
            # Clean up the entire temp directory tree
            temp_root = os.path.dirname(model_dir)
            if temp_root.startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_root)
                logger.info(f"Cleaned up temporary directory: {temp_root}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {e}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    load_models()
    logger.info("Application startup complete")

# In-memory sermon storage
# Structure: {mosque_name: {"segments": [{"text": arabic_text, "translations": {"en": eng_trans, "ur": ur_trans}}], "timestamp": last_update_time}}
sermon_storage: Dict[str, Dict] = {}

# Translation cache
# Structure: {mosque_name: {lang: {arabic_text: translated_text}}}
translation_cache: Dict[str, Dict[str, Dict[str, str]]] = {}

# Version tracking for cleaning on new app version
APP_VERSION = None  # Will be set by /app_version endpoint
last_seen_version = None

# Define the request models
class TranslationRequest(BaseModel):
    text: str
    target_lang: str  # e.g. "en", "ur", etc.
    mosque_name: str  # Added to support per-mosque caching

class SermonSegmentRequest(BaseModel):
    mosque_name: str
    segment: str

# Health check endpoint for EB
@app.get("/health")
async def health_check():
    """Health check endpoint for AWS Elastic Beanstalk"""
    return {
        "status": "healthy",
        "models_loaded": model is not None and tokenizer is not None,
        "urdu_models_loaded": urdu_model is not None and urdu_tokenizer is not None
    }

# Helper: Clean sermons not updated for 3 hours (10800 seconds)
def clean_old_sermons():
    now = int(time.time())
    to_delete = []
    for mosque, data in list(sermon_storage.items()):
        last_update = data.get("timestamp", 0) // 1000  # ms to s
        if now - last_update > 10800:  # 3 hours
            to_delete.append(mosque)
    for mosque in to_delete:
        logger.info(f"Auto-cleaning old sermon for mosque: {mosque}")
        del sermon_storage[mosque]

# Helper: Normalize Arabic text for cache and translation
def normalize_arabic(text):
    # Remove Arabic diacritics, collapse spaces, trim, remove invisible chars
    text = re.sub(r'[\u064B-\u0652]', '', text)  # Remove Arabic diacritics
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u200C', '').replace('\u200D', '')
    return text.strip()

@app.post("/app_version")
async def app_version(request: Request):
    global APP_VERSION, last_seen_version, sermon_storage
    data = await request.json()
    version = data.get("version")
    if version is None:
        return {"cleared": False, "reason": "No version provided"}
    if last_seen_version != version:
        logger.info(f"App version changed from {last_seen_version} to {version}. Clearing all sermons.")
        sermon_storage.clear()
        last_seen_version = version
        APP_VERSION = version
        return {"cleared": True}
    return {"cleared": False}

@app.get("/")
async def root():
    clean_old_sermons()
    return {"message": "Translation API is running on AWS Elastic Beanstalk"}

@app.post("/translate")
async def translate(req: TranslationRequest):
    clean_old_sermons()
    try:
        # Check if models are loaded
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
            
        logger.info(f"Received translation request: {req.text[:50]}... -> {req.target_lang}")

        # Use custom model for English, base model for Urdu
        local_tokenizer = tokenizer
        local_model = model

        if req.target_lang == "ur":
            if urdu_model is None or urdu_tokenizer is None:
                raise HTTPException(status_code=503, detail="Urdu model not loaded yet")
            logger.info("Using base M2M100 model for Urdu translation")
            local_tokenizer = urdu_tokenizer
            local_model = urdu_model

        # Check if text is empty
        if not req.text or req.text.strip() == "":
            return JSONResponse(
                content={"translated": ""},
                media_type="application/json; charset=utf-8"
            )

        # Normalize the text
        normalized_text = normalize_arabic(req.text)

        # Check if translation exists in cache
        if req.mosque_name in translation_cache:
            if req.target_lang in translation_cache[req.mosque_name]:
                if normalized_text in translation_cache[req.mosque_name][req.target_lang]:
                    logger.info(f"Using cached translation for {req.mosque_name} - {req.target_lang}")
                    return JSONResponse(
                        content={"translated": translation_cache[req.mosque_name][req.target_lang][normalized_text]},
                        media_type="application/json; charset=utf-8"
                    )

        # Set source language to Arabic
        local_tokenizer.src_lang = "ar"

        # Ensure target language is valid
        if req.target_lang not in ["en", "ur"]:
            logger.warning(f"Unsupported target language: {req.target_lang}. Defaulting to English.")
            req.target_lang = "en"

        # Configure tokenizer for the target language
        try:
            bos = local_tokenizer.lang_code_to_id[req.target_lang]
        except KeyError:
            logger.warning(f"Language code {req.target_lang} not found in tokenizer. Using English.")
            bos = local_tokenizer.lang_code_to_id["en"]

        # Tokenize the input text
        enc = local_tokenizer(normalized_text, return_tensors="pt")

        # Generate translation
        generated = local_model.generate(
            **enc,
            forced_bos_token_id=bos,
            max_length=150,
            num_beams=5,
            length_penalty=0.8,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        # Decode the generated tokens
        out = local_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        logger.info(f"Translation complete. Result: {out[:50]}...")

        # Store in cache
        if req.mosque_name not in translation_cache:
            translation_cache[req.mosque_name] = {}
        if req.target_lang not in translation_cache[req.mosque_name]:
            translation_cache[req.mosque_name][req.target_lang] = {}
        translation_cache[req.mosque_name][req.target_lang][normalized_text] = out

        # Return the translation with proper encoding
        return JSONResponse(
            content={"translated": out},
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/add_sermon_segment")
async def add_sermon_segment(req: SermonSegmentRequest):
    """Add a new segment to a mosque's sermon with translations"""
    clean_old_sermons()
    try:
        # Check if models are loaded
        if model is None or tokenizer is None or urdu_model is None or urdu_tokenizer is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
            
        mosque_name = req.mosque_name
        segment = req.segment
        logger.info(f"[DEBUG] Adding sermon segment for mosque: {mosque_name}")
        logger.info(f"[DEBUG] Segment content: {segment[:50]}...")
        
        # Initialize if this mosque isn't in storage yet
        if mosque_name not in sermon_storage:
            sermon_storage[mosque_name] = {
                "segments": [],
                "timestamp": int(time.time() * 1000)
            }
        
        # Generate translations for the segment
        translations = {}
        for lang in ["en", "ur"]:
            try:
                if lang == "ur":
                    # Use the base model for Urdu
                    local_tokenizer = urdu_tokenizer
                    local_model = urdu_model
                else:
                    # Use the fine-tuned model for English
                    local_tokenizer = tokenizer
                    local_model = model
                local_tokenizer.src_lang = "ar"
                try:
                    bos = local_tokenizer.lang_code_to_id[lang]
                except KeyError:
                    logger.warning(f"Language code {lang} not found in tokenizer. Using English.")
                    bos = local_tokenizer.lang_code_to_id["en"]
                enc = local_tokenizer(segment, return_tensors="pt")
                generated = local_model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    max_length=150,
                    num_beams=5,
                    length_penalty=0.8,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                translated = local_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                translations[lang] = translated
            except Exception as e:
                logger.error(f"Translation error for {lang}: {str(e)}")
                translations[lang] = ""
        # Add the new segment with translations and update timestamp
        sermon_storage[mosque_name]["segments"].append({
            "text": segment,
            "translations": translations
        })
        sermon_storage[mosque_name]["timestamp"] = int(time.time() * 1000)
        
        logger.info(f"[DEBUG] Current segments for {mosque_name}: {sermon_storage[mosque_name]['segments']}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error adding sermon segment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding sermon segment: {str(e)}")

@app.get("/get_sermon_updates")
async def get_sermon_updates(mosque: str, since: int = 0):
    """Get updates for a mosque's sermon since the specified timestamp"""
    clean_old_sermons()
    try:
        logger.info(f"Fetching sermon updates for mosque: {mosque}, since: {since}")
        
        # Return empty response if mosque not in storage
        if mosque not in sermon_storage:
            return JSONResponse(
                content={
                    "has_updates": False,
                    "segments": [],
                    "timestamp": int(time.time() * 1000)
                }
            )
        
        current_timestamp = sermon_storage[mosque]["timestamp"]
        
        # If the client's timestamp is older than our latest update,
        # return all segments (for simplicity)
        if since < current_timestamp:
            return JSONResponse(
                content={
                    "has_updates": True,
                    "segments": sermon_storage[mosque]["segments"],
                    "timestamp": current_timestamp
                }
            )
        else:
            # No new updates
            return JSONResponse(
                content={
                    "has_updates": False,
                    "segments": [],
                    "timestamp": current_timestamp
                }
            )
    except Exception as e:
        logger.error(f"Error getting sermon updates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sermon updates: {str(e)}")

@app.get("/get_sermon")
async def get_sermon(mosque: str):
    """Get all segments for a mosque's sermon with translations"""
    clean_old_sermons()
    try:
        logger.info(f"[DEBUG] get_sermon called for mosque: {mosque}")
        if mosque not in sermon_storage:
            logger.info(f"[DEBUG] No segments found for {mosque}")
            return JSONResponse(
                content={"segments": []},
                media_type="application/json; charset=utf-8"
            )
        logger.info(f"[DEBUG] Returning segments for {mosque}: {sermon_storage[mosque]['segments']}")
        return JSONResponse(
            content={"segments": sermon_storage[mosque]["segments"]},
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error getting sermon: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sermon: {str(e)}")

@app.delete("/clear_sermon")
async def clear_sermon(mosque: str):
    """Clear all sermon content and translations for a specific mosque"""
    try:
        logger.info(f"[DEBUG] clear_sermon called for mosque: {mosque}")
        
        has_data = False
        
        if mosque in sermon_storage:
            del sermon_storage[mosque]
            has_data = True

        if mosque in translation_cache:
            del translation_cache[mosque]
            has_data = True

        if has_data:
            logger.info(f"[DEBUG] Cleared sermon and translations for {mosque}")
            return {
                "success": True,
                "message": f"Sermon and translations for {mosque} have been cleared"
            }
        else:
            logger.info(f"[DEBUG] No sermon data found for {mosque}, skipping clear.")
            return {
                "success": True,
                "message": f"No sermon data found for {mosque}"
            }

    except Exception as e:
        logger.error(f"Error clearing sermon: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing sermon: {str(e)}")

@app.get("/active_mosques")
async def get_active_mosques():
    """Get a list of all mosques with active sermons"""
    clean_old_sermons()
    try:
        active_mosques = list(sermon_storage.keys())
        return {"mosques": active_mosques}
    except Exception as e:
        logger.error(f"Error getting active mosques: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active mosques: {str(e)}")

# For EB deployment - expose the app
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)