import json
import requests
import io
from pypdf import PdfReader
from deep_translator import GoogleTranslator
import ftfy
import unicodedata


def process_text(text):
    # 1. Clean
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)

    # 2. Translate (German -> English)
    # Check if text is not empty
    if text.strip():
        try:
            # Using Google Translate (Free wrapper)
            # Note: For production, use DeepL API for better German accuracy
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            return text  # Fallback to original
    return text

def extract_pdf_to_json(pdf_url):
    # 1. Fetch the PDF file
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return json.dumps({"error": f"Failed to download PDF: {e}"})

    # Wrap the content in a bytes buffer so pypdf can read it
    pdf_file = io.BytesIO(response.content)

    # 2. Initialize PDF Reader
    reader = PdfReader(pdf_file)

    # 3. Attempt to get the document title from metadata
    # (PDFs rarely have specific "page titles", so we use the document title)
    doc_title = "Untitled"
    if reader.metadata and reader.metadata.title:
        doc_title = reader.metadata.title

    output_data = []

    # 4. Loop through every page
    for page in reader.pages:
        text = page.extract_text()

        # Clean up text (optional: remove excessive whitespace)
        # text = " ".join(text.split())

        page_entry = {
            "type": "pdf",
            "page": page.page_number,
            "url": pdf_url,
            "title": doc_title,
            "content": process_text(text)
        }

        output_data.append(page_entry)

    # 5. Return as JSON string
    with open("tuwien_informatics_pdf_demo.json", "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


# --- Usage Example ---
# (Using a sample PDF URL for demonstration)
url = "https://informatics.tuwien.ac.at/master/curriculum-ue066937.pdf"
json_output = extract_pdf_to_json(url)
