import os
import requests
import json
import tempfile
import pymupdf4llm
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from deep_translator import GoogleTranslator

os.environ["GROQ_API_KEY"] = "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"


class StudyMetadata(BaseModel):
    program_name: str = Field(description="Name of the study program, e.g., 'Master Software Engineering'")
    degree_level: str = Field(description="Degree level: 'Bachelor', 'Master'")
    ects: Optional[int] = Field(description="Total ECTS credits")
    language: str = Field(description="Main language of the document")
    document_type: str = Field(description="Type of document: 'Curriculum', 'Guideline'")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(StudyMetadata)

def translate_text(text):
    translator = GoogleTranslator(source='auto', target='en')
    if not text.strip(): return ""
    try:
        if len(text) > 4000:
            print("Text longer than 4000 characters")
            return translator.translate(text[:4000]) + "..."
        return translator.translate(text)
    except Exception as e:
        print(f"Translation warning: {e}")
        return text

def get_pdf_metadata_with_llm(text):
    system_prompt = "Extract metadata from this university document text."
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")])
    chain = prompt | structured_llm
    try:
        return chain.invoke({"text": text[:3000]}).model_dump()
    except:
        return {}

def clean_page_content(text, program_name):
    lines = text.split('\n')
    
    start_index = 0
    for i in range(min(3, len(lines))):
        line = lines[i].strip().lower()
        prog = program_name.lower()
        if len(line) < 5 or line in prog or prog in line:
            start_index += 1
        else:
            break
            
    return "\n".join(lines[start_index:])

def process_pdf_smart(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except Exception as e:
        print(f" Failed to download: {e}")
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf.flush()

        md_data = pymupdf4llm.to_markdown(temp_pdf.name, page_chunks=True)

    if not md_data: return []

    first_page_text = md_data[0]["text"]
    global_metadata = get_pdf_metadata_with_llm(first_page_text)
    global_metadata["source_url"] = pdf_url
    
    print(f"   Detected Program: {global_metadata.get('program_name')}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, 
        separators=["\n\n", "\n", ".", " ", ""]
    )

    final_output = []
    total_pages = len(md_data)

    
    for i, page in enumerate(md_data):
        raw_text = page["text"]
        page_num = page["metadata"]["page"]
        
        if not raw_text.strip(): continue
        
        print(f"   Processing Page {page_num}/{total_pages}...", end="\r")

        clean_text = clean_page_content(raw_text, global_metadata.get('program_name', ''))
        
        if not clean_text.strip(): continue

        translated_text = translate_text(clean_text)

        chunks = text_splitter.create_documents([translated_text])
        
        for chunk in chunks:
            entry = {
                "content": chunk.page_content,
                "metadata": {
                    **global_metadata,
                    "page_number": page_num, 
                    "chunk_strategy": "page_recursive"
                }
            }
            final_output.append(entry)

    print(f"\nDone! Processed {len(final_output)} chunks.")
    return final_output

if __name__ == "__main__":
    url = "https://informatics.tuwien.ac.at/master/curriculum-ue066935.pdf"
    data = process_pdf_smart(url)
    
    if data:
        with open("tuwien_media_human_centered_computing.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)