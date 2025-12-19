
import os
import argparse
import sys
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_comic.content_data import PAGES
from transformer_comic.image_generator import ImageGenerator, BananaGenerator, GeminiGenerator
from transformer_comic.html_generator import HTMLGenerator
from transformer_comic.pdf_converter import PDFConverter

def main():
    load_dotenv() # Load .env if present

    parser = argparse.ArgumentParser(description="Generate a Transformer Technical Comic Book")
    
    # Provider selection
    parser.add_argument("--provider", choices=['google', 'banana', 'gemini'], default='gemini', help="Image generation provider. 'gemini' for Nano Banana/AIza key.")
    
    # Auth Args
    parser.add_argument("--api-key", help="API Key (Google 'AIza...' or other)", default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--project-id", help="Google Cloud Project ID (Vertex only)", default=os.getenv("GOOGLE_PROJECT_ID"))
    parser.add_argument("--banana-model-key", help="Banana.dev Model Key", default=os.getenv("BANANA_MODEL_KEY"))
    
    # Common Args
    parser.add_argument("--output", help="Output directory", default="output")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and use placeholders")
    parser.add_argument("--skip-images", action="store_true", help="Skip image generation entirely (use existing if available)")
    
    args = parser.parse_args()
    
    # Setup directories
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print("Welcome to the Transformer Comic Generator!")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"Provider: {args.provider.upper()}")
    
    # 1. Initialize Image Generator
    img_gen = None
    
    if args.provider == 'google':
        # Vertex AI (Classic)
        img_gen = ImageGenerator(project_id=args.project_id or "my-project", dry_run=args.dry_run)
    elif args.provider == 'banana':
        # Banana.dev
        img_gen = BananaGenerator(
            api_key=args.api_key, # Reusing generic api-key arg
            model_key=args.banana_model_key,
            dry_run=args.dry_run
        )
    elif args.provider == 'gemini':
        # Google Generative AI (Nano Banana)
        img_gen = GeminiGenerator(
            api_key=args.api_key,
            dry_run=args.dry_run
        )
    
    # 2. Generate Images
    print("\n--- Phase 1: Image Generation ---")
    
    generated_pages = []
    
    for page in tqdm(PAGES, desc="Processing Pages"):
        image_filename = f"page_{page['page_number']:02d}.png"
        image_path = os.path.join(images_dir, image_filename)
        
        # Prepare page object for HTML rendering
        page_data = page.copy()
        page_data['image_path'] = f"images/{image_filename}" # Relative path
        
        if not args.skip_images and img_gen:
            success = img_gen.generate_image(
                prompt=page['image_prompt'],
                output_path=image_path
            )
            if not success:
               # Warn but continue (placeholder likely generated)
               pass
                
        generated_pages.append(page_data)
        
    # 3. Generate HTML
    print("\n--- Phase 2: HTML Composition ---")
    html_gen = HTMLGenerator(template_dir=os.path.join(os.path.dirname(__file__), 'transformer_comic', 'templates'))
    html_output_path = os.path.join(args.output, "transformer_comic.html")
    
    if html_gen.generate_html(generated_pages, html_output_path):
        print(f"Success! HTML generated at: {html_output_path}")
    else:
        print("Failed to generate HTML.")
        return

    # 4. Convert to PDF
    print("\n--- Phase 3: PDF Conversion ---")
    pdf_conv = PDFConverter()
    pdf_output_path = os.path.join(args.output, "transformer_comic.pdf")
    
    if pdf_conv.convert(html_output_path, pdf_output_path):
        print(f"Success! PDF generated at: {pdf_output_path}")
    else:
        print("Failed to generate PDF (WeasyPrint might not be fully configured).")

    print("\n--- Process Complete! ---")
    print(f"You can view your comic at: file://{os.path.abspath(html_output_path)}")

if __name__ == "__main__":
    main()
