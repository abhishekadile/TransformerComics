
# Transformer Comic Generator

This tool generates a 23-page comic book-style technical document explaining Transformers, from embeddings to attention mechanisms.

## Prerequisites

1. **Python 3.8+**
2. **Google Cloud Credentials** (for image generation) - *Optional if running in dry-run mode*
   - Enable Vertex AI API
   - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: WeasyPrint requires additional system dependencies (GTK3). Check [WeasyPrint docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation) if you encounter errors.*

2. (Optional) Create a `.env` file with your credentials:
   ```
   GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
   ```

## Usage

### 1. Dry Run (Fastest, uses placeholders)
Generates the HTML/PDF with blue placeholder images. Perfect for testing layout.
```bash
python generate_comic.py --dry-run
```

### 2. Full Generation
Requires Google Cloud Project ID.
```bash
python generate_comic.py --project-id YOUR_PROJECT_ID
```

### 3. Skip Image Generation
If you've already generated images and just want to update the text/HTML.
```bash
python generate_comic.py --skip-images
```

## Output

Check the `/output` directory for:
- `transformer_comic.html`: Interactive single-page web version
- `transformer_comic.pdf`: Print-ready PDF file
- `images/`: The generated images

## Customization

Edit `transformer_comic/content_data.py` to change the narratives, prompts, or technical explanations.
