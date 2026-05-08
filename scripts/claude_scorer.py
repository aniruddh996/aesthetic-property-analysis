# claude code to rank the images from zip file
#!/usr/bin/env python3
"""
Rate 800 images (400 urban + 400 landscape) on Alexander's 15 Fundamental Properties
Using Claude API

Author: Aniruddh
Date: 2026-03-15
"""

import anthropic
import base64
import os
import pandas as pd
from pathlib import Path
import time
import json
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
API_KEY = #insert api key

# Paths to your image folders
URBAN_DIR = "urban_400"  # Update this to your urban images folder path
LANDSCAPE_DIR = "landscape_400"  # Update this to your landscape images folder path

# Output files
OUTPUT_CSV = "all_images_15_properties.csv"
PROGRESS_FILE = "rating_progress.csv"

# API settings
MODEL = "claude-sonnet-4-20250514"
RATE_LIMIT_DELAY = 1  # seconds between API calls

# =========================
# ALEXANDER'S 15 PROPERTIES
# =========================
PROPERTIES = [
    "levels_of_scale",
    "strong_centers",
    "boundaries",
    "alternating_repetition",
    "positive_space",
    "good_shape",
    "local_symmetries",
    "deep_interlock_and_ambiguity",
    "contrast",
    "gradients",
    "roughness",
    "echoes",
    "the_void",
    "simplicity_and_inner_calm",
    "not_separateness"
]

PROPERTY_DESCRIPTIONS = {
    "levels_of_scale": "Multiple scales present, from large structures to fine details",
    "strong_centers": "Clear focal points or centers of attention that organize the space",
    "boundaries": "Well-defined edges and transitions between different areas",
    "alternating_repetition": "Repeating elements that alternate in a rhythmic pattern",
    "positive_space": "Spaces and voids that are well-shaped and contribute positively",
    "good_shape": "Pleasing, coherent forms and shapes throughout",
    "local_symmetries": "Small-scale symmetries within parts of the image",
    "deep_interlock_and_ambiguity": "Elements that interweave and create rich visual complexity",
    "contrast": "Strong contrasts in light, color, texture, or form",
    "gradients": "Smooth transitions and gradations rather than abrupt changes",
    "roughness": "Natural variation and irregularity rather than perfect uniformity",
    "echoes": "Similar shapes or patterns that echo and reinforce each other",
    "the_void": "Empty or calm spaces that provide visual rest",
    "simplicity_and_inner_calm": "Overall sense of peace, order, and simplicity",
    "not_separateness": "Elements feel connected and unified with their surroundings"
}

# =========================
# HELPER FUNCTIONS
# =========================
def encode_image(image_path):
    """Convert image to base64 for API"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_image_media_type(image_path):
    """Determine image media type"""
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(ext, 'image/jpeg')

def create_rating_prompt():
    """Create the prompt for rating images on 15 properties"""
    
    properties_list = "\n".join([
        f"{i+1}. **{prop.replace('_', ' ').title()}**: {PROPERTY_DESCRIPTIONS[prop]}"
        for i, prop in enumerate(PROPERTIES)
    ])
    
    prompt = f"""You are an expert in architectural aesthetics and Christopher Alexander's pattern language theory.

Analyze this image and rate it on each of Christopher Alexander's 15 Fundamental Properties of Wholeness. These properties describe what makes spaces feel alive, beautiful, and whole.

**The 15 Properties:**
{properties_list}

**Instructions:**
1. Carefully examine the image
2. Rate each property on a scale of 1-10:
   - 1-3: Property is absent or weak
   - 4-6: Property is moderately present
   - 7-9: Property is strongly present
   - 10: Property is exceptionally present

3. Provide your ratings in JSON format ONLY (no additional text):

{{
    "levels_of_scale": X,
    "strong_centers": X,
    "boundaries": X,
    "alternating_repetition": X,
    "positive_space": X,
    "good_shape": X,
    "local_symmetries": X,
    "deep_interlock_and_ambiguity": X,
    "contrast": X,
    "gradients": X,
    "roughness": X,
    "echoes": X,
    "the_void": X,
    "simplicity_and_inner_calm": X,
    "not_separateness": X
}}

Respond ONLY with the JSON object, no other text."""

    return prompt

def rate_image(client, image_path, max_retries=3):
    """Rate a single image using Claude API"""
    
    # Encode image
    image_data = encode_image(image_path)
    media_type = get_image_media_type(image_path)
    
    # Create prompt
    prompt = create_rating_prompt()
    
    # Try with retries
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            # Extract JSON response
            response_text = message.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            ratings = json.loads(response_text)
            
            return ratings
            
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"    Failed after {max_retries} attempts")
                return None
    
    return None

def get_all_images():
    """Get all image paths from both folders"""
    images = []
    
    # Get urban images
    urban_path = Path(URBAN_DIR)
    if urban_path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in urban_path.glob(ext):
                images.append({
                    'path': str(img_path),
                    'image_name': img_path.name,
                    'image_type': 'urban'
                })
    
    # Get landscape images
    landscape_path = Path(LANDSCAPE_DIR)
    if landscape_path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in landscape_path.glob(ext):
                images.append({
                    'path': str(img_path),
                    'image_name': img_path.name,
                    'image_type': 'landscape'
                })
    
    return images

def load_progress():
    """Load previously rated images to resume"""
    if os.path.exists(PROGRESS_FILE):
        try:
            df = pd.read_csv(PROGRESS_FILE)
            completed = set(df['image_name'].tolist())
            print(f"  Loaded progress: {len(completed)} images already rated")
            return df, completed
        except:
            pass
    return pd.DataFrame(), set()

def save_progress(results):
    """Save progress periodically"""
    df = pd.DataFrame(results)
    df.to_csv(PROGRESS_FILE, index=False)

# =========================
# MAIN PROCESSING
# =========================
def main():
    print("="*60)
    print("RATING IMAGES ON ALEXANDER'S 15 PROPERTIES")
    print("="*60)
    
    # Initialize client
    print("\nInitializing Claude API client...")
    client = anthropic.Anthropic(api_key=API_KEY)
    
    # Get all images
    print("Scanning image folders...")
    all_images = get_all_images()
    
    urban_count = sum(1 for img in all_images if img['image_type'] == 'urban')
    landscape_count = sum(1 for img in all_images if img['image_type'] == 'landscape')
    
    print(f"\nFound images:")
    print(f"  Urban: {urban_count}")
    print(f"  Landscape: {landscape_count}")
    print(f"  Total: {len(all_images)}")
    
    if len(all_images) == 0:
        print("\n❌ ERROR: No images found!")
        print(f"   Check that folders exist:")
        print(f"   - {URBAN_DIR}")
        print(f"   - {LANDSCAPE_DIR}")
        return
    
    # Load progress
    print("\nChecking for previous progress...")
    existing_df, completed_images = load_progress()
    
    # Filter out already completed images
    images_to_rate = [img for img in all_images if img['image_name'] not in completed_images]
    
    print(f"\nImages to rate: {len(images_to_rate)}")
    if len(images_to_rate) == 0:
        print("✓ All images already rated!")
        return
    
    # Start rating
    print("\n" + "="*60)
    print("STARTING RATING PROCESS")
    print("="*60)
    print(f"Rate limit delay: {RATE_LIMIT_DELAY} seconds between requests")
    print(f"Estimated time: {len(images_to_rate) * RATE_LIMIT_DELAY / 60:.1f} minutes")
    print("")
    
    # Convert existing results to list
    if len(existing_df) > 0:
        results = existing_df.to_dict('records')
    else:
        results = []
    
    # Process each image
    for i, img_info in enumerate(tqdm(images_to_rate, desc="Rating images"), 1):
        image_path = img_info['path']
        image_name = img_info['image_name']
        image_type = img_info['image_type']
        
        # Rate the image
        ratings = rate_image(client, image_path)
        
        if ratings:
            result = {
                'image_name': image_name,
                'image_type': image_type
            }
            result.update(ratings)
            results.append(result)
            
            # Save progress every 10 images
            if i % 10 == 0:
                save_progress(results)
                tqdm.write(f"  Progress saved: {len(results)} images completed")
        else:
            tqdm.write(f"  ✗ Failed to rate: {image_name}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save final results
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)
    
    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Complete!")
    print(f"  Total images rated: {len(results)}")
    print(f"  Urban: {len([r for r in results if r['image_type'] == 'urban'])}")
    print(f"  Landscape: {len([r for r in results if r['image_type'] == 'landscape'])}")
    print(f"\n  Results saved to: {OUTPUT_CSV}")
    
    # Show statistics
    print("\n" + "="*60)
    print("RATING STATISTICS")
    print("="*60)
    
    for prop in PROPERTIES:
        if prop in df_final.columns:
            mean = df_final[prop].mean()
            std = df_final[prop].std()
            print(f"  {prop:35s}: Mean={mean:.2f}, Std={std:.2f}")
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
