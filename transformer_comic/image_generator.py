
import os
import time
import logging
import base64
from io import BytesIO
from typing import Optional

# Google Cloud Imports (Vertex AI)
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
except ImportError:
    pass

# Google Generative AI Imports (Gemini / Nano Banana)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Banana Dev Imports
try:
    import banana_dev as banana
except ImportError:
    pass

import requests
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class BaseGenerator:
    def _generate_placeholder(self, prompt: str, output_path: str, error_msg: str = "") -> bool:
        try:
            img = Image.new('RGB', (1024, 1024), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "Placeholder Image", fill=(255,255,0))
            d.text((10,50), prompt[:100], fill=(255,255,255))
            d.text((10,90), f"(Generator Failed: {error_msg})", fill=(255,100,100))
            img.save(output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to create placeholder: {e}")
            return False

class ImageGenerator(BaseGenerator):
    """Legacy/Advanced Vertex AI Generator"""
    def __init__(self, project_id: str, location: str = "us-central1", dry_run: bool = False):
        self.dry_run = dry_run
        self.project_id = project_id
        
        if not dry_run:
            try:
                aiplatform.init(project=project_id, location=location)
                self.model = aiplatform.ImageGenerationModel.from_pretrained("imagegeneration@006")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}. Switching to dry_run mode.")
                self.dry_run = True

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = "") -> bool:
        if os.path.exists(output_path):
            logger.info(f"Image already exists at {output_path}, skipping.")
            return True

        if self.dry_run:
            logger.info(f"[Dry Run - Vertex] Generating placeholder for: {prompt[:30]}...")
            return self._generate_placeholder(prompt, output_path)

        for attempt in range(3):
            try:
                logger.info(f"Generating image (Vertex attempt {attempt+1}): {prompt[:30]}...")
                response = self.model.generate_images(
                    prompt=prompt + ", vibrant comic book art style, bold colors, clean lines, technical diagrams with comic book aesthetic",
                    number_of_images=1,
                    guidance_scale=7.5,
                    aspect_ratio="1:1", 
                    negative_prompt=negative_prompt or "blurry, low quality, pixelated, watermark, text, ugly, deformed"
                )
                if response and response.images:
                    response.images[0].save(output_path)
                    logger.info(f"Saved image to {output_path}")
                    return True
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                time.sleep(2 ** attempt)
        
        return self._generate_placeholder(prompt, output_path, "Vertex API Failed")

class GeminiGenerator(BaseGenerator):
    """Google Generative AI (Nano Banana/Gemini) with API Key"""
    def __init__(self, api_key: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = api_key
        
        if not api_key:
            logger.warning("Google API Key missing. Switching to dry_run mode.")
            self.dry_run = True
            return

        try:
            genai.configure(api_key=api_key)
            self.model_name = 'gemini-3-pro-image-preview' # Nano Banana official ID if available
            # Alternate: 'gemini-2.5-flash-image-preview'
        except Exception as e:
            logger.error(f"Failed to configure GenAI: {e}")
            self.dry_run = True

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = "") -> bool:
        if os.path.exists(output_path):
            logger.info(f"Image already exists at {output_path}, skipping.")
            return True

        if self.dry_run:
            logger.info(f"[Dry Run - Gemini] Generating placeholder for: {prompt[:30]}...")
            return self._generate_placeholder(prompt, output_path)

        full_prompt = prompt + ", vibrant comic book art style, bold colors, clean lines, technical diagrams with comic book aesthetic"

        for attempt in range(3):
            try:
                logger.info(f"Generating image (Gemini attempt {attempt+1}): {prompt[:30]}...")
                
                model = genai.GenerativeModel(self.model_name)
                
                # As per user instruction for Gemini Image generation
                response = model.generate_content(full_prompt)
                
                # Accessing image data based on user snippet provided
                # "image = response.candidates[0].content.parts[0].inline_data"
                if response and response.candidates and response.candidates[0].content.parts:
                    part = response.candidates[0].content.parts[0]
                    
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # inline_data usually has .data (bytes) and .mime_type
                        image_data = part.inline_data.data 
                        
                        img = Image.open(BytesIO(image_data))
                        img.save(output_path)
                        logger.info(f"Saved image to {output_path}")
                        return True
                    else:
                        # Fallback parsing if structure differs slightly
                        # Sometimes it might be directly in a different property if SDK updates
                        logger.warning(f"Response structure unexpected: {part}")
                
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                time.sleep(2 ** attempt)

        logger.error("GenAI generation failed.")
        return self._generate_placeholder(prompt, output_path, "GenAI API Error")

class BananaGenerator(BaseGenerator):
    """Banana.dev Generator"""
    def __init__(self, api_key: str, model_key: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = api_key
        self.model_key = model_key
        
        if not api_key or not model_key:
            logger.warning("Banana API Key or Model Key missing. Switching to dry_run mode.")
            self.dry_run = True

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = "") -> bool:
        if os.path.exists(output_path):
            logger.info(f"Image already exists at {output_path}, skipping.")
            return True

        if self.dry_run:
            logger.info(f"[Dry Run - Banana] Generating placeholder for: {prompt[:30]}...")
            return self._generate_placeholder(prompt, output_path)

        model_inputs = {
            "prompt": prompt + ", vibrant comic book art style, bold colors, clean lines, technical diagrams with comic book aesthetic",
            "negative_prompt": negative_prompt or "blurry, low quality, pixelated, watermark, text, ugly, deformed",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024
        }

        for attempt in range(3):
            try:
                logger.info(f"Generating image (Banana attempt {attempt+1}): {prompt[:30]}...")
                out = banana.run(self.api_key, self.model_key, model_inputs)
                if 'modelOutputs' in out and len(out['modelOutputs']) > 0:
                    image_data = out['modelOutputs'][0].get('image_base64') or out['modelOutputs'][0].get('image')
                    if image_data:
                        image_bytes = base64.b64decode(image_data)
                        img = Image.open(BytesIO(image_bytes))
                        img.save(output_path)
                        logger.info(f"Saved image to {output_path}")
                        return True
            except Exception as e:
                logger.error(f"Error generating image (Banana): {e}")
                time.sleep(2 ** attempt)
        
        return self._generate_placeholder(prompt, output_path, "Banana API Failed")
