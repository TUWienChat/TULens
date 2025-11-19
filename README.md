# TUWien Informatics Chat

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Web scraping
A Scrapy spider for crawling and extracting content from the TU Wien Informatics website.
### Usage

```bash
scrapy crawl tuwien_informatics_spider.py -o output.json -a download_dir_path="downloads"
````

### Output Structure

The spider generates two types of items:

#### HTML Pages
```json
{
  "type": "html",
  "url": "https://...",
  "title": "Page Title",
  "headings": "H1 H2 H3 content...",
  "text": "Paragraph and list content...",
  "emails": ["email@example.com"]
}
```
#### Binary Files
```json
{
  "type": "file",
  "content_type": "application/pdf",
  "url": "https://...",
  "saved_as": "downloads/pdf/filename-uuid.pdf",
  "size_bytes": 12345
}
```

Files are organized into subdirectories: `pdf/`, `images/`, `zip/`, `other/`.