"""
PDF document loader for extracting text from PDF files.
"""
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pypdf

class PDFLoader:
    """
    Loader for PDF documents.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF loader.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load a PDF document and split it into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with text and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from PDF
        text = self._extract_text(file_path)
        
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Create document chunks with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return documents
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of text
            end = min(start + self.chunk_size, len(text))
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for a period, question mark, or exclamation point followed by whitespace
                for i in range(end - 1, max(start, end - 100), -1):
                    if text[i] in ['.', '!', '?'] and (i + 1 >= len(text) or text[i + 1].isspace()):
                        end = i + 1
                        break
            
            # Add chunk to list
            chunks.append(text[start:end])
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def load_documents(self, directory: Union[str, Path], recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Load all PDF documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            List of document chunks with text and metadata
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        documents = []
        
        # Get all PDF files in the directory
        if recursive:
            pdf_files = list(directory.glob("**/*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        # Load each PDF file
        for pdf_file in pdf_files:
            try:
                doc_chunks = self.load_document(pdf_file)
                documents.extend(doc_chunks)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        return documents