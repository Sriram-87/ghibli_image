import os
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image
import io
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import requests
from tqdm.auto import tqdm

class GhibliStyler:
    def _init_(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
    def download_if_needed(self, url, filename):
        """Download file if it doesn't exist locally"""
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(filename, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, 
                    desc=filename
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"Downloaded {filename}")
            
    def setup_model(self):
        """Set up the Stable Diffusion model with optimal Ghibli settings"""
        print("Setting up the Ghibli transformation pipeline...")
        
        # Use DDIM scheduler for better quality and control
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        
        # First try to load the specialized Ghibli model
        try:
            model_id = "nitrosocke/Ghibli-Diffusion"
            
            # Load with optimal settings for Ghibli aesthetics
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                scheduler=scheduler,
                safety_checker=None
            )
            
            # Move to GPU if available 
            self.pipe.to(self.device)
            
            # Optimize memory usage
            self.pipe.enable_attention_slicing()
            
            if self.device == "cuda":
                # Enable memory efficient attention if available for better quality
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("Using xformers for memory efficient attention")
                except:
                    print("xformers not available, using default attention")
            
            print("Specialized Ghibli model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading specialized model: {e}")
            print("Falling back to standard Stable Diffusion model...")
            
            # Fallback to standard model if specialized one fails
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                scheduler=scheduler,
                safety_checker=None
            )
            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            print("Standard model loaded as fallback.")
        
        return self.pipe
        
    def preprocess_image(self, image):
        """Prepare image for optimal Ghibli transformation"""
        # Convert to RGB format
        image = image.convert("RGB")
        
        # Calculate dimensions that work optimally for diffusion models
        width, height = image.size
        aspect_ratio = width / height
        
        # Size for optimal quality while respecting aspect ratio
        if aspect_ratio > 1:  # Landscape
            new_width = min(768, int(512 * aspect_ratio))
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait or square
            new_height = min(768, int(512 / aspect_ratio)) 
            new_width = int(new_height * aspect_ratio)
        
        # SD works best with dimensions divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Resize with high quality resampling
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Normalize colors for better transformation
        img_array = np.array(resized_img)
        normalized_img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        
        return Image.fromarray(normalized_img)
    
    def build_ghibli_prompt(self, image_type="scene"):
        """Build an optimal prompt for authentic Ghibli aesthetics"""
        # Base Ghibli aesthetics that apply to all images
        ghibli_base = "Studio Ghibli style, hand-painted, Miyazaki masterpiece"
        
        # Specialized prompt components
        aesthetics = [
            "soft watercolor textures",
            "painterly brushstrokes",
            "gentle pastel palette", 
            "hazy atmospheric perspective",
            "detailed background art",
            "cinematic lighting"
        ]
        
        # Image type specific components
        type_specific = {
            "scene": [
                "nostalgic fantasy world",
                "magical rural landscape", 
                "dreamy pastoral scenery",
                "idyllic natural setting"
            ],
            "character": [
                "whimsical character design",
                "expressive facial features",
                "enchanting character pose",
                "character with Ghibli proportions"
            ],
            "portrait": [
                "charming portrait",
                "soulful eyes",
                "gentle expression",
                "character with warm presence"
            ]
        }
        
        # Select a subset of aesthetics to keep prompt focused
        chosen_aesthetics = np.random.choice(aesthetics, 3, replace=False)
        chosen_type_specifics = np.random.choice(type_specific.get(image_type, type_specific["scene"]), 2, replace=False)
        
        # Assemble full prompt
        prompt_parts = [ghibli_base] + list(chosen_aesthetics) + list(chosen_type_specifics)
        prompt = ", ".join(prompt_parts)
        
        # Essential negative prompt to avoid common issues
        negative_prompt = "photorealistic, 3D, low quality, blurry, grainy, oversaturated, ugly, distorted features, anime eyes too large, deformed, disfigured, bad anatomy, deep fried, meme"
        
        return prompt, negative_prompt
    
    def generate_ghibli_image(self, image, strength=0.68, guidance_scale=7.5, seed=None, image_type="scene"):
        """Generate an authentic Ghibli-style image"""
        if self.pipe is None:
            self.setup_model()
            
        # Prepare the image
        processed_img = self.preprocess_image(image)
        
        # Get optimized prompt
        prompt, negative_prompt = self.build_ghibli_prompt(image_type)
        print(f"Using prompt: {prompt}")
        
        # Configure generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        print("Transforming image to Ghibli style...")
        start_time = time.time()
        
        # Generate with optimal parameters for Ghibli style
        result = self.pipe(
            prompt=prompt,
            image=processed_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=75,  # Higher step count for better quality
            negative_prompt=negative_prompt,
            generator=generator
        ).images[0]
        
        print(f"Base transformation completed in {time.time() - start_time:.2f} seconds")
        
        # Apply post-processing for authentic Ghibli look
        enhanced_result = self.apply_ghibli_post_processing(result)
        
        return processed_img, result, enhanced_result
    
    def apply_ghibli_post_processing(self, image):
        """Apply post-processing effects to match Ghibli's distinctive look"""
        # Convert to array for manipulation
        img_array = np.array(image)
        
        # 1. Convert to BGR for OpenCV operations
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 2. Apply very slight bilateral filter for the signature Ghibli soft-yet-detailed look
        bilateral = cv2.bilateralFilter(bgr, 9, 17, 17)
        
        # 3. Enhance colors to match Ghibli palette
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
        
        # Subtle saturation adjustment
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255).astype(np.uint8)
        
        # 4. Subtle brightness adjustment for that watercolor impression
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.07, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        color_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 5. Apply a very slight watercolor-like effect
        # First convert to Lab color space for better manipulation
        lab = cv2.cvtColor(color_adjusted, cv2.COLOR_BGR2Lab)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply subtle blurring to a and b channels (color info) while preserving L (details)
        a = cv2.GaussianBlur(a, (3, 3), 0.5)
        b = cv2.GaussianBlur(b, (3, 3), 0.5)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        watercolor = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        # 6. Very subtle edge preservation for the hand-drawn feel
        edges = cv2.Canny(l, 100, 200)
        edges = cv2.dilate(edges, None)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend edge information
        result = cv2.addWeighted(watercolor, 0.92, edges_3channel, 0.08, 0)
        
        # Convert back to RGB for PIL
        rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(rgb_result)

def run_ghibli_conversion():
    """Main function to run the Ghibli image converter"""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = "✅ GPU is available!" if device == "cuda" else "⚠ Warning: GPU not available. Processing will be slow."
    print(gpu_info)
    
    # Initialize the GhibliStyler
    styler = GhibliStyler()
    
    # Setup the model
    styler.setup_model()
    
    # Upload image section
    print("\n📁 Please upload your image file:")
    uploaded = files.upload()
    
    if uploaded:
        file_name = list(uploaded.keys())[0]
        image = Image.open(io.BytesIO(uploaded[file_name]))
        
        # Display original image
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()
        
        # Ask for image type
        print("\n🖼 Select image type for better results:")
        print("1: Scene/Landscape")
        print("2: Character")
        print("3: Portrait")
        image_type_input = input("Enter choice (1-3, default: 1): ").strip()
        
        image_type_map = {
            "1": "scene",
            "2": "character", 
            "3": "portrait"
        }
        image_type = image_type_map.get(image_type_input, "scene")
        
        # Ask for strength input with error handling
        while True:
            try:
                strength = float(input("\n🔄 Enter stylization strength (0.5-0.75, recommended 0.68): "))
                strength = max(0.5, min(0.75, strength))
                break
            except ValueError:
                print("Invalid input. Please enter a number between 0.5 and 0.75.")
        
        # Ask for prompt strength
        while True:
            try:
                guidance_scale = float(input("\n🎯 Enter prompt guidance strength (7.0-9.0, recommended 7.5): "))
                guidance_scale = max(7.0, min(9.0, guidance_scale))
                break
            except ValueError:
                print("Invalid input. Please enter a number between 7.0 and 9.0.")
        
        # Optional: Set random seed for reproducibility
        seed_input = input("\n🎲 Enter a seed number for reproducibility (leave blank for random): ")
        seed = int(seed_input) if seed_input.strip() else None
        
        print("\n🎬 Starting Ghibli transformation process...")
        
        # Generate and display the result
        processed_img, basic_result, enhanced_result = styler.generate_ghibli_image(
            image, 
            strength=strength, 
            guidance_scale=guidance_scale, 
            seed=seed,
            image_type=image_type
        )
        
        # Display side by side comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(processed_img)
        plt.title("Prepared Input")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(basic_result)
        plt.title("Base Ghibli Style")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(enhanced_result)
        plt.title("Perfect Ghibli Style")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the output image and offer download
        output_filename = f"perfect_ghibli_{file_name}"
        enhanced_result.save(output_filename)
        files.download(output_filename)
        print(f"\n✅ Image saved as {output_filename} and download initiated!")
        
        # Provide information about the conversion
        print("\n📊 Transformation Information:")
        print(f"- Image Type: {image_type}")
        print(f"- Stylization Strength: {strength}")
        print(f"- Prompt Guidance: {guidance_scale}")
        print(f"- Seed: {seed if seed is not None else 'Random'}")
        
        print("\n✨ Transformation complete! If you want to try different settings, run the cell again.")
    else:
        print("No file was uploaded. Please run the cell again and upload an image.")

# Execute the transformation process
if _name_ == "_main_":
    print("🎨 Studio Ghibli Perfect Image Transformer 🎨")
    print("This tool will transform your photos into authentic Studio Ghibli style artwork.")
    run_ghibli_conversion()
