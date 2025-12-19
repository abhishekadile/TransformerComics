
import os
import logging
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class HTMLGenerator:
    def __init__(self, template_dir: str):
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_html(self, pages: list, output_path: str):
        try:
            template = self.env.get_template('comic_page.html')
            
            # Process pages to ensure image paths are relative for HTML
            rendered_html = template.render(pages=pages)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
                
            logger.info(f"Generated HTML at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate HTML: {e}")
            return False
