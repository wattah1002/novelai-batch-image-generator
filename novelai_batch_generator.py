#!/usr/bin/env python3
import asyncio
import aiohttp
import aiofiles
import json
import os
import sys
import argparse
import base64
import zipfile
import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
import random
import logging

# Load environment variables
load_dotenv()

# NovelAI API configuration
NOVELAI_API_URL = "https://api.novelai.net"
IMAGE_API_URL = "https://image.novelai.net"

# Available models
MODELS = {
    "NAI_DIFFUSION_4_5_CURATED_PREVIEW": "nai-diffusion-4-5-curated-preview",
    "NAI_DIFFUSION_4_CURATED_PREVIEW": "nai-diffusion-4-curated-preview",
    "NAI_DIFFUSION_3": "nai-diffusion-3",
    "NAI_DIFFUSION_3_INPAINTING": "nai-diffusion-3-inpainting",
    "NAI_DIFFUSION": "nai-diffusion",
    "NAI_DIFFUSION_2": "nai-diffusion-2",
    "SAFE_DIFFUSION": "safe-diffusion",
    "NAI_DIFFUSION_FURRY": "nai-diffusion-furry"
}

# Default configuration
DEFAULT_CONFIG = {
    "model": "NAI_DIFFUSION_3",
    "width": 512,
    "height": 768,
    "steps": 28,
    "scale": 11.0,
    "sampler": "k_euler_ancestral",
    "sm": False,
    "sm_dyn": False,
    "dynamic_thresholding": False,
    "controlnet_strength": 1.0,
    "legacy": False,
    "add_original_image": False,
    "ucPreset": 0,
    "qualityToggle": True
}

class NovelAIBatchGenerator:
    def __init__(self, token: str):
        self.token = token
        self.session = None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def generate_image(self, prompt: str, negative_prompt: str, config: Dict[str, Any]) -> Optional[bytes]:
        """Generate a single image using NovelAI API"""
        async with self.semaphore:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "accept": "application/x-zip-compressed"
            }
            
            model_key = config.get("model", "NAI_DIFFUSION_3")
            model_value = MODELS.get(model_key, MODELS["NAI_DIFFUSION_3"])
            
            payload = {
                "input": prompt,
                "model": model_value,
                "action": "generate",
                "parameters": {
                    "width": config.get("width", 512),
                    "height": config.get("height", 768),
                    "scale": config.get("scale", 11.0),
                    "sampler": config.get("sampler", "k_euler_ancestral"),
                    "steps": config.get("steps", 28),
                    "seed": random.randint(0, 4294967295),
                    "n_samples": 1,
                    "ucPreset": config.get("ucPreset", 0),
                    "qualityToggle": config.get("qualityToggle", True),
                    "sm": config.get("sm", False),
                    "sm_dyn": config.get("sm_dyn", False),
                    "dynamic_thresholding": config.get("dynamic_thresholding", False),
                    "controlnet_strength": config.get("controlnet_strength", 1.0),
                    "legacy": config.get("legacy", False),
                    "add_original_image": config.get("add_original_image", False),
                    "negative_prompt": negative_prompt
                }
            }
            
            try:
                async with self.session.post(
                    f"{IMAGE_API_URL}/ai/generate-image",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        zip_data = await response.read()
                        # Extract image from zip
                        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                            # Get the first image file in the zip
                            for file_info in zf.filelist:
                                if file_info.filename.endswith('.png'):
                                    return zf.read(file_info.filename)
                        return None
                    else:
                        error_text = await response.text()
                        logging.error(f"API Error {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                logging.error("Request timeout")
                return None
            except Exception as e:
                logging.error(f"Generation error: {str(e)}")
                return None
                
    async def generate_batch(self, prompts: List[str], negative_prompts: List[str], 
                           count: int, config: Dict[str, Any], output_dir: Path):
        """Generate multiple images in batch"""
        tasks = []
        
        # Create progress bar
        pbar = async_tqdm(total=count, desc="Generating images")
        
        # Prepare prompts for generation
        prompt_list = []
        for i in range(count):
            prompt = random.choice(prompts) if len(prompts) > 1 else prompts[0]
            negative_prompt = random.choice(negative_prompts) if len(negative_prompts) > 1 else negative_prompts[0]
            prompt_list.append((prompt, negative_prompt, i))
            
        async def generate_and_save(prompt: str, negative_prompt: str, index: int):
            image_data = await self.generate_image(prompt, negative_prompt, config)
            if image_data:
                filename = output_dir / "images" / f"image_{index:04d}.png"
                async with aiofiles.open(filename, 'wb') as f:
                    await f.write(image_data)
                logging.info(f"Saved: {filename}")
            else:
                logging.error(f"Failed to generate image {index}")
            pbar.update(1)
            
        # Create tasks for all generations
        for prompt, negative_prompt, i in prompt_list:
            task = generate_and_save(prompt, negative_prompt, i)
            tasks.append(task)
            
        # Execute all tasks
        await asyncio.gather(*tasks)
        pbar.close()

def load_prompts(file_path: str) -> List[str]:
    """Load prompts from a text file"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {file_path}")
        sys.exit(1)
    
    if not prompts:
        logging.error(f"No prompts found in file: {file_path}")
        sys.exit(1)
        
    return prompts

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file: {config_path}")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_generation_info(output_dir: Path, prompts: List[str], negative_prompts: List[str], 
                        config: Dict[str, Any], args: argparse.Namespace):
    """Save generation information to output directory"""
    # Save configuration
    config_file = output_dir / "generation_config.json"
    full_config = {
        "command_args": vars(args),
        "api_config": config,
        "timestamp": datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(full_config, f, indent=2)
        
    # Save prompts used
    prompts_file = output_dir / "prompts_used.txt"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts))
        
    # Save negative prompts used
    negative_prompts_file = output_dir / "negative_prompts_used.txt"
    with open(negative_prompts_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(negative_prompts))

async def main():
    parser = argparse.ArgumentParser(description="NovelAI Batch Image Generator")
    parser.add_argument("--prompt-file", default="prompts.txt", help="Path to prompts file")
    parser.add_argument("--negative-prompt-file", default="negative_prompts.txt", help="Path to negative prompts file")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--steps", type=int, help="Sampling steps")
    parser.add_argument("--scale", type=float, help="Guidance scale")
    parser.add_argument("--sampler", help="Sampler to use")
    parser.add_argument("--output-dir", help="Custom output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get API token
    token = os.getenv("NOVELAI_TOKEN")
    if not token:
        logging.error("NOVELAI_TOKEN not found in environment variables")
        logging.error("Please create a .env file with NOVELAI_TOKEN=your_token_here")
        sys.exit(1)
        
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    negative_prompts = load_prompts(args.negative_prompt_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config["model"] = args.model
    if args.width:
        config["width"] = args.width
    if args.height:
        config["height"] = args.height
    if args.steps:
        config["steps"] = args.steps
    if args.scale:
        config["scale"] = args.scale
    if args.sampler:
        config["sampler"] = args.sampler
        
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("output") / f"batch_{timestamp}"
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generation info
    save_generation_info(output_dir, prompts, negative_prompts, config, args)
    
    # Setup log file
    log_file = output_dir / "generation_log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log generation start
    logging.info(f"Starting batch generation of {args.count} images")
    logging.info(f"Model: {config['model']}")
    logging.info(f"Resolution: {config['width']}x{config['height']}")
    logging.info(f"Output directory: {output_dir}")
    
    # Generate images
    async with NovelAIBatchGenerator(token) as generator:
        await generator.generate_batch(prompts, negative_prompts, args.count, config, output_dir)
        
    logging.info(f"Generation complete! Images saved to: {output_dir}")
    print(f"\nGeneration complete! Images saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())