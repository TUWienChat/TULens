import os
import uuid
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from markdownify import markdownify as md


class TuwienInformaticsSpider(CrawlSpider):
    name = "tuwien_informatics_msc"
    allowed_domains = ["informatics.tuwien.ac.at"]
    start_urls = ["https://informatics.tuwien.ac.at/master"]  # Focused starting point

    custom_settings = {
        'DEPTH_LIMIT': 2,  # Limits how many clicks away from start_urls we go
        'DEPTH_PRIORITY': 1,  # Combined with next setting for BFO
        'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
        'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
    }

    def __init__(self, download_dir_path="downloads", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_dir_path = download_dir_path

    # Define rules to filter and follow links
    rules = (
        # RULE 1: Only follow links related to studies and admission
        # We ignore 'news', 'events', and 'staff' to keep the RAG focused
        Rule(
            LinkExtractor(
                allow=(r'/study/', r'/admission/', r'/master/', r'/people/',),
                deny=(r'/news/', r'/events/', r'/research/'),
                unique=True
            ),
            callback='parse_item',
            follow=True
        ),
        # RULE 2: Download PDFs directly without following links inside them
        Rule(
            LinkExtractor(
                allow=(r'.*\.pdf',),
                deny_extensions=[],
                unique=True,
                tags=('a', 'area'),
                attrs=('href',)
            ),
            callback='handle_binary_response',
            follow=False
        ),
    )

    def parse_item(self, response):
        main_content = response.css("main").get() or response.css("body").get()

        # Convert HTML to structured Markdown
        markdown_text = md(main_content, heading_style="ATX", strip=['a', 'img'])

        # Extract breadcrumbs for context (e.g., Bachelor > Admission)
        breadcrumbs = response.css(".breadcrumb-item ::text").getall()

        yield {
            "type": "html",
            "url": response.url,
            "title": response.css("title::text").get(default="").strip(),
            "breadcrumb": " > ".join([b.strip() for b in breadcrumbs if b.strip()]),
            "content": markdown_text,
            "metadata": {
                "language": response.xpath("/html/@lang").get(),
                "depth": response.meta.get("depth")
            }
        }

    def handle_binary_response(self, response):
        """Saves PDFs and other files to the local disk."""
        content_type = response.headers.get("Content-Type", b"").decode().lower()

        # Determine subfolder
        if "pdf" in content_type:
            folder = "pdf"
            ext = "pdf"
        else:
            folder = "other"
            ext = "bin"

        filename = f"{uuid.uuid4()}.{ext}"
        download_dir = os.path.join(self.download_dir_path, folder)
        os.makedirs(download_dir, exist_ok=True)
        filepath = os.path.join(download_dir, filename)

        with open(filepath, "wb") as f:
            f.write(response.body)

        yield {
            "type": "file",
            "url": response.url,
            "saved_as": filepath,
            "content_type": content_type
        }