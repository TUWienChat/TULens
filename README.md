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
scrapy crawl tuwien_informatics_msc -o output.json -a download_dir_path="downloads"
````

### Output Structure

The spider generates two types of items:

#### HTML Pages
```json
{
  "type": "html",
  "url": "https://...",
  "title": "Page Title",
  "breadcrumb": "H1 H2 H3 content...",
  "content": "Paragraph and list content..."
}
```
#### Binary Files
```json
{
  "type": "file",
  "url": "https://...",
  "saved_as": "downloads/pdf/filename-uuid.pdf",
  "content_type": "application/pdf"
}
```