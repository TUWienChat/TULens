from langchain_community.document_loaders import RecursiveUrlLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import re
import os
import requests
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse
import hashlib
from dotenv import load_dotenv

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, CHROMADB_COLLECTION

#load_dotenv()

CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", 8000)
URLS_PATH = "scrape_urls.json"

# --- CONFIGURATION ---
# BASE_URLS = [
#     {
#         "url": "https://informatics.tuwien.ac.at/master/",
#         "pattern": r"https://informatics.tuwien.ac.at/master/(data-science|business-informatics|software-engineering|embedded-computing-systems|logic-and-artificial-intelligence|media-and-human-centered-computing|medical-informatics|visual-computing)/.*",
#         "depth": 2
#     },
#     {
#         "url": "https://www.tuwien.at/en/studies/admission/",
#         "pattern": r"https://www.tuwien.at/en/studies/admission/(masters-programmes|academic-calendar|changing-your-degree-programme)/.+",
#         "depth": 2
#     },
#     {
#         "url": "https://tiss.tuwien.ac.at/hilfe/zu/tiss/faq",
#         "pattern": None,
#         "depth": 1
#     },
#     {
#         "url": "https://tiss.tuwien.ac.at/hilfe/zu/tiss/education",
#         "pattern": None,
#         "depth": 1
#     },
#     {
#         "url": "https://tiss.tuwien.ac.at/hilfe/zu/tiss/organisation",
#         "pattern": None,
#         "depth": 1
#     },
#     {
#         "url": "https://tiss.tuwien.ac.at/hilfe/zu/tiss/new_erste_info_stud",
#         "pattern": None,
#         "depth": 1
#     }
# ]

def build_patterns_from_json(config_path=URLS_PATH):
    # 1. Load the JSON data
    with open(config_path, 'r') as file:
        data = json.load(file)

    processed_urls = []

    # 2. Iterate and rebuild patterns
    for entry in data:
        base_url = entry['url']  # Remove trailing slash
        suffix = entry.get('suffix', '')  # Default suffix if missing
        subpages = entry.get('pages_filter', [])
        pattern = entry.get('pattern', None)
        depth = entry.get('depth', 1)

        if subpages:
            # Escape strings to ensure special chars don't break regex
            # Join them: (page1|page2|page3)
            group_pattern = "|".join([re.escape(p) for p in subpages])

            # Construct: base + /(group) + suffix
            pattern_str = f"{re.escape(base_url)}({group_pattern})/{suffix}"
            pattern_str = re.compile(pattern_str).pattern
        elif pattern:
            # Fallback: If no subpages, just append suffix to url
            pattern_str = pattern
        else:
            pattern_str = None

        # 3. Store the compiled object
        processed_urls.append({
            "url": base_url,
            "pattern": pattern_str,
            "depth": depth
        })

    return processed_urls

BASE_URLS = build_patterns_from_json()

def clean_pdf_text(text: str) -> str:
    """Enhanced PDF text cleaning with better formatting preservation."""
    # 1. Fix line-break hyphenation (e.g., "Infor- \n matics" -> "Informatics")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 2. Preserve bullet points and list structures
    text = re.sub(r"•", "\n• ", text)
    text = re.sub(r"[▪▫]", "\n• ", text)

    # 3. Remove multiple newlines but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Remove non-printable characters
    text = text.replace('\x0c', '')
    text = text.replace('\u200b', '')  # Zero-width space

    # 5. Fix common PDF artifacts
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation

    # 6. Normalize whitespace
    text = re.sub(r" +", " ", text).strip()

    return text


def extract_metadata_from_url(url: str) -> dict:
    """Extract meaningful metadata from URL structure."""
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]

    metadata = {
        'domain': parsed.netloc,
        'path_depth': len(path_parts)
    }

    # Extract program name if in informatics URL
    if 'informatics.tuwien.ac.at' in url:
        for part in path_parts:
            if part in ['data-science', 'business-informatics', 'software-engineering',
                        'embedded-computing-systems', 'logic-and-artificial-intelligence',
                        'media-and-human-centered-computing', 'medical-informatics', 'visual-computing']:
                metadata['program'] = part.replace('-', ' ').title()
                break

    # Identify content type from URL
    if 'admission' in url:
        metadata['content_type'] = 'admission'
    elif 'curriculum' in url:
        metadata['content_type'] = 'curriculum'
    elif 'course' in url:
        metadata['content_type'] = 'course'
    else:
        metadata['content_type'] = 'general'

    return metadata


def extract_pdf(doc: Document, processed_pdfs: set) -> list[Document]:
    """Enhanced PDF extraction with better error handling and metadata."""
    pdf_documents = []

    try:
        response = requests.get(doc.metadata['source'], timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        for link in soup.find_all('a', href=re.compile(r'.*\.pdf$')):
            pdf_url = link['href']

            # Normalize PDF URL
            if not pdf_url.startswith('http'):
                pdf_url = urljoin(doc.metadata['source'], pdf_url)

            if pdf_url in processed_pdfs:
                continue

            print(f"Loading PDF: {pdf_url}")

            try:
                loader = PyPDFLoader(pdf_url)
                temp_docs = loader.load()

                # Extract program name from parent page
                program_name = doc.metadata.get('program', 'Unknown')

                for pdf_doc in temp_docs:
                    # Clean the text
                    pdf_doc.page_content = clean_pdf_text(pdf_doc.page_content)

                    # Enhanced metadata
                    pdf_doc.metadata['type'] = 'pdf'
                    pdf_doc.metadata['content_type'] = 'curriculum'
                    pdf_doc.metadata['program'] = program_name
                    pdf_doc.metadata['parent_url'] = doc.metadata['source']
                    pdf_doc.metadata['pdf_url'] = pdf_url

                    # Add title extraction from link text
                    link_text = link.get_text(strip=True)
                    if link_text:
                        pdf_doc.metadata['title'] = link_text

                pdf_documents.extend(temp_docs)
                processed_pdfs.add(pdf_url)

            except Exception as e:
                print(f"Failed to load PDF {pdf_url}: {e}")

    except Exception as e:
        print(f"Error processing page {doc.metadata['source']}: {e}")

    return pdf_documents


def html_extractor(html: str) -> str:
    """Enhanced HTML extraction with better content preservation."""
    soup = BeautifulSoup(html, "lxml")

    # Remove navigation, footers, scripts, and other "noise"
    unwanted = ["nav", "footer", "header", "aside", "script", "style",
                "form", "button", "iframe", "noscript"]
    for element in soup(unwanted):
        element.decompose()

    # Remove common noise classes/ids
    noise_patterns = ['cookie', 'advertisement', 'social-share', 'breadcrumb']
    for pattern in noise_patterns:
        for element in soup.find_all(class_=re.compile(pattern, re.I)):
            element.decompose()
        for element in soup.find_all(id=re.compile(pattern, re.I)):
            element.decompose()

    # Convert to Markdown
    markdown_content = md(str(soup), heading_style="ATX", strip=['img', 'video'])

    # Clean up markdown
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
    markdown_content = re.sub(r"\[]\(.*?\)", "", markdown_content)  # Remove empty links
    markdown_content = re.sub(r"^\s*[-*_]{3,}\s*$", "", markdown_content, flags=re.MULTILINE)  # Remove horizontal rules

    return markdown_content.strip()


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Remove duplicate documents based on content hash."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        # Create hash of content
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    removed = len(documents) - len(unique_docs)
    if removed > 0:
        print(f"Removed {removed} duplicate documents")

    return unique_docs


def enrich_document_metadata(doc: Document) -> Document:
    """Add additional metadata to documents for better retrieval."""
    # Extract URL metadata
    url_metadata = extract_metadata_from_url(doc.metadata.get('source', ''))
    doc.metadata.update(url_metadata)

    # Add content length
    doc.metadata['content_length'] = len(doc.page_content)

    # Extract potential title from content
    if 'title' not in doc.metadata:
        lines = doc.page_content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if line.startswith('#'):
                doc.metadata['title'] = line.lstrip('#').strip()
                break

    return doc


# --- MAIN INGESTION PIPELINE ---
print("=" * 80)
print("STARTING DOCUMENT INGESTION PIPELINE")
print("=" * 80)

# STEP 1: Scrape Web Pages
print("\n[1/6] Starting recursive web scrape...")
web_docs = []
for base_url in BASE_URLS:
    print(f"  Scraping: {base_url['url']}")
    web_loader = RecursiveUrlLoader(
        url=base_url['url'],
        max_depth=base_url['depth'],
        extractor=html_extractor,
        link_regex=base_url['pattern'],
        prevent_outside=True
    )
    web_docs.extend(web_loader.load())

print(f"  ✓ Scraped {len(web_docs)} web pages")

# STEP 2: Enrich web document metadata
print("\n[2/6] Enriching metadata...")
web_docs = [enrich_document_metadata(doc) for doc in web_docs]
print(f"  ✓ Enhanced metadata for {len(web_docs)} documents")

# STEP 3: Extract PDFs
print("\n[3/6] Identifying and loading curriculum PDFs...")
pdf_docs = []
processed_pdfs = set()

for doc in web_docs:
    extracted_pdfs = extract_pdf(doc, processed_pdfs)
    if extracted_pdfs:
        pdf_docs.extend(extracted_pdfs)

print(f"  ✓ Loaded {len(pdf_docs)} PDF pages from {len(processed_pdfs)} PDFs")

# STEP 4: Combine and deduplicate
print("\n[4/6] Combining and deduplicating documents...")
all_documents = web_docs + pdf_docs
all_documents = deduplicate_documents(all_documents)
print(f"  ✓ Total unique documents: {len(all_documents)}")

# STEP 5: Split documents into chunks
print("\n[5/6] Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""]  # Better semantic splitting
)

all_splits = text_splitter.split_documents(all_documents)
print(f"  ✓ Created {len(all_splits)} chunks")

# STEP 6: Store in vector database
print("\n[6/6] Storing in vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

# Delete existing collection to avoid duplicates
try:
    client.delete_collection(CHROMADB_COLLECTION)
    print("  Deleted existing collection")
except:
    pass

vectorstore = Chroma.from_documents(
    client=client,
    documents=all_splits,
    embedding=embeddings,
    collection_name=CHROMADB_COLLECTION
)

print(f"  ✓ Successfully ingested {len(all_splits)} chunks")

# STEP 7: Print summary statistics
print("\n" + "=" * 80)
print("INGESTION COMPLETE - SUMMARY")
print("=" * 80)
print(f"Web pages scraped: {len(web_docs)}")
print(f"PDFs processed: {len(processed_pdfs)}")
print(f"PDF pages extracted: {len(pdf_docs)}")
print(f"Total documents: {len(all_documents)}")
print(f"Total chunks stored: {len(all_splits)}")
print(f"Average chunk size: {sum(len(doc.page_content) for doc in all_splits) / len(all_splits):.0f} chars")
print("=" * 80)