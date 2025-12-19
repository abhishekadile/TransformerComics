
import logging
import os

try:
    from weasyprint import HTML, CSS
except ImportError:
    pass

logger = logging.getLogger(__name__)

class PDFConverter:
    def convert(self, html_path: str, output_path: str):
        if 'HTML' not in globals():
            logger.error("WeasyPrint not installed or failed to load. Cannot generate PDF.")
            return False

        try:
            logger.info(f"Converting {html_path} to PDF...")
            # Ideally verify weasyprint is installed
            
            # Base URL is required for relative paths (images) to resolve correctly
            base_url = os.path.dirname(html_path)
            
            HTML(html_path, base_url=base_url).write_pdf(output_path)
            
            logger.info(f"Generated PDF at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            return False
