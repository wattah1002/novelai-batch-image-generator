# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NovelAI batch image generator that uses the NovelAI API to generate multiple images asynchronously. The main script (`novelai_batch_generator.py`) reads prompts from text files and generates images based on configuration settings.

## Key Architecture

- **NovelAIBatchGenerator class**: Core async class that handles API communication and image generation
  - Uses aiohttp for async HTTP requests with rate limiting (max 5 concurrent connections)
  - Implements retry logic with exponential backoff for failed requests
  - Saves images with metadata (prompt in EXIF data)

## Environment Setup

1. **API Token**: Requires `NOVELAI_TOKEN` in `.env` file
2. **Virtual Environment**: Uses venv (already created in `venv/` directory)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic image generation (100 images with default settings)
python novelai_batch_generator.py

# Generate with custom settings
python novelai_batch_generator.py --count 50 --model NAI_DIFFUSION_4_5_CURATED --steps 23

# Run with custom batch name
python novelai_batch_generator.py --batch-name "test_run" --count 10
```

## Input/Output Structure

- **Input Files**:
  - `prompts.txt`: One prompt per line (required)
  - `negative_prompts.txt`: One negative prompt per line (optional)
  - `config.json`: Default generation parameters

- **Output Structure**: `output/batch_YYYYMMDD_HHMMSS/`
  - `images/`: Generated PNG files with naming pattern `{batch_name}_prompt{index}_seed{seed}.png`
  - `generation_config.json`: Actual configuration used
  - `generation_log.txt`: Detailed generation log
  - `prompts_used.txt` & `negative_prompts_used.txt`: Actual prompts used

## Configuration

Default config.json parameters:

- model: NAI_DIFFUSION_4_5_FULL
- width: 832, height: 1216
- steps: 23, scale: 5.0
- sampler: k_euler_ancestral

Available models:

- NAI_DIFFUSION_4_5_FULL (default)
- NAI_DIFFUSION_4_5_CURATED

## Error Handling

The generator implements:

- Automatic retry with exponential backoff (up to 3 attempts)
- Detailed error logging to generation_log.txt
- Skips failed images and continues with remaining prompts
- HTTP 429 (rate limit) handling with longer delays
