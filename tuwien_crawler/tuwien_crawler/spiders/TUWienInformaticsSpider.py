import scrapy
import os
import uuid
from markdownify import markdownify as md

class TuwienInformaticsSpider(scrapy.Spider):
    name = "tuwien_informatics"
    allowed_domains = ["informatics.tuwien.ac.at"]
    start_urls = ["https://informatics.tuwien.ac.at/master/"]

    def __init__(self, download_dir_path="downloads", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_dir_path = download_dir_path

    def __clean_text(self, text):
        text = text.replace("\n", " ").replace("\t", " ")
        return text.strip()

    def __clean_list(self, text_list):
        clean_list = [self.__clean_text(text) for text in text_list]
        return self.__clean_text(" ".join(clean_list))


    def __handle_binary_response(self, response):
        url = response.url

        # detect content type
        content_type = response.headers.get("Content-Type", b"").decode().lower()

        # choose folder based on type
        if "pdf" in content_type:
            content_dir = "pdf"
            extension = "pdf"
        elif "image/png" in content_type:
            content_dir = "images"
            extension = "jpg"
        elif "image/jpeg" in content_type:
            content_dir = "images"
            extension = "png"
        elif "image/svg+xml" in content_type:
            content_dir = "images"
            extension = "svg"
        elif "zip" in content_type:
            content_dir = "zip"
            extension = "zip"
        else:
            content_dir = "other"
            extension = "bin"

        if not extension == "pdf":
            return

        filename = url.split("/")[-1].split(".")[0] # URL ends with /
        filename = f"{filename}-{uuid.uuid4()}.{extension}"

        download_dir = f"{self.download_dir_path}/{content_dir}"
        os.makedirs(download_dir, exist_ok=True)
        filepath = f"{download_dir }/{filename}"

        # write file
        with open(filepath, "wb") as f:
            f.write(response.body)

        yield {
            "type": "file",
            "content_type": content_type,
            "url": url,
            "saved_as": filepath,
            "size_bytes": len(response.body),
        }

    def __handle_html_response(self, response):
        # 1. Target the main content area to avoid nav/footer noise
        main_content = response.css("main").get()
        if not main_content:
            main_content = response.css("body").get()

        # 2. Convert to Markdown to preserve structure for the LLM
        # We strip links to avoid "click here" noise, but keep headers and lists
        markdown_text = md(main_content, heading_style="ATX", strip=['a', 'img'])

        # 3. Extract Breadcrumbs for hierarchical context
        breadcrumbs = response.css(".breadcrumb-item ::text").getall()
        breadcrumb_path = " > ".join([b.strip() for b in breadcrumbs if b.strip()])

        yield {
            "type": "html",
            "url": response.url,
            "title": response.css("title::text").get(default="").strip(),
            "breadcrumb": breadcrumb_path,
            "content": markdown_text,
            "metadata": {
                "language": response.xpath("/html/@lang").get(),
                "scraped_at": scrapy.utils.project.get_project_settings().get("TIMESTAMP"),  # Useful for RAG versioning
            }
        }

        # 4. Smart Link Following
        # Use a more restrictive approach to stay on enrollment-relevant pages
        for link in response.css("a::attr(href)").getall():
            url = response.urljoin(link)
            # Only follow internal links and ignore fragments (#)
            if url.startswith("https://informatics.tuwien.ac.at") and "#" not in url:
                yield response.follow(url, callback=self.parse)
        # main = response.css("main")
        #
        # emails = [
        #     href.replace("mailto:", "")
        #     for href in main.css("a::attr(href)").getall()
        #     if href.startswith("mailto:")
        # ]
        #
        # yield {
        #     "type": "html",
        #     "url": response.url,
        #     "title": self.__clean_text(main.css("title::text").get(default="")),
        #     "headings": self.__clean_list(main.css("h1::text, h2::text, h3::text").getall()),
        #     "text": self.__clean_list(main.css("p::text, li::text, td::text").getall()),
        #     "emails": emails
        # }
        #
        # links = response.css("a::attr(href)").getall()
        # for link in links:
        #     url = response.urljoin(link)
        #     if url.startswith("mailto:"):
        #         continue
        #     if any(domain in url for domain in self.allowed_domains):
        #         yield response.follow(url, callback=self.parse)


    def parse(self, response):
        if isinstance(response, scrapy.http.HtmlResponse):
            yield from self.__handle_html_response(response)
        elif isinstance(response, scrapy.http.Response):
            yield from self.__handle_binary_response(response)