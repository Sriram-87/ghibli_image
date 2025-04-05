# Studio Ghibli Image Transformer

This project uses **Stable Diffusion** and **AI** to transform your photos into **Studio Ghibli-style** art. The model works by processing your image through an AI pipeline and applying post-processing to give it that signature Ghibli aesthetic—soft textures, vibrant colors, and detailed backgrounds.

## Privacy & Data Security

- **No data storage**: The images are processed temporarily during the session and are **never stored**.
- Your **privacy is protected**: No personal data is collected or shared.

## How It Works

1. **Upload Your Photo**: Choose a **scene**, **character**, or **portrait** style.
2. **AI Transformation**: The AI processes the image and applies the Ghibli style.
3. **Post-processing**: The image is refined with subtle effects to make it look like hand-painted art.
4. **Download**: Your transformed artwork is ready in just a few minutes!

## How to Use

1. Clone or download this repository to your local machine.
2. Set up a Python environment with the required dependencies (see below).
3. Run the script using **Google Colab** or a **Jupyter Notebook**.
4. Upload an image, select the transformation style, and adjust the settings (strength, guidance).
5. Download your **Studio Ghibli-style artwork**!

### Requirements

- Python 3.x
- `torch`
- `diffusers`
- `PIL`
- `requests`
- `tqdm`
- `opencv-python` (for post-processing effects)

### Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/ghibli-image-transformer.git
