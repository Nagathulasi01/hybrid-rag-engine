import os
import logging
import pdfplumber
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def parse_txt(file_path: str) -> str:
    """
    Extract text from a TXT file.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        str: The contents of the text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_and_chunk(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, str]]:
    """
    Parses a document (PDF or TXT) and splits it into chunks.
    
    Args:
        file_path (str): Path to the uploaded file.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks to preserve context.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing chunk text and metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        full_text = parse_pdf(file_path)
    elif ext == '.txt':
        full_text = parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    # We use RecursiveCharacterTextSplitter to ensure semantic splits where possible
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    
    chunks = text_splitter.split_text(full_text)
    logger.info(f"Split {file_path} into {len(chunks)} chunks.")
    
    # Return structured dicts
    return [{"content": chunk, "metadata": {"source": os.path.basename(file_path)}} for chunk in chunks]
