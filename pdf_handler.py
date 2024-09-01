from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embeddings
import pypdfium2

def get_pdf_texts(pdfs_bytes):
    """
        Extracts text from a list of PDF files provided as bytes.

        This function takes a list of PDF files in byte format and extracts the text
        content from each PDF using the `extract_text_from_pdf` function.

        Args:
            pdfs_bytes (list of bytes): A list where each element is a PDF file represented
                as bytes.

        Returns:
            list of str: A list of strings, where each string contains the extracted text
            from a corresponding PDF file.
    """
    return [extract_text_from_pdf(pdfs_byte) for pdfs_byte in pdfs_bytes]

def extract_text_from_pdf(pdf_bytes):
    """
       Extracts text from a PDF file provided as bytes.

       This function takes a PDF file in byte format, loads it using the PyPDFium2 library,
       and extracts text from each page. The extracted text from all pages is combined
       into a single string with each page's content separated by newline characters.

       Args:
           pdf_bytes (bytes): The PDF file represented as bytes.

       Returns:
           str: A string containing the extracted text from all pages of the PDF.
    """
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))

def get_text_chunks(text):
    """
        Splits a large text into smaller chunks for processing.

        This function takes a large string of text and splits it into smaller chunks
        using the `RecursiveCharacterTextSplitter`. The splitting is done based on
        specified separators (like newlines) and allows for some overlap between chunks
        to maintain context.

        Args:
            text (str): The large text string to be split into smaller chunks.

        Returns:
            list of str: A list of text chunks, where each chunk is a substring of the input text.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, separators=["\n", "\n\n"])
    return splitter.split_text(text)

def get_document_chunks(text_list):
    """
        Converts a list of text strings into a list of document chunks.

        This function takes a list of large text strings and splits each text into smaller
        chunks using the `get_text_chunks` function. Each chunk is then wrapped in a `Document`
        object to create a list of document chunks.

        Args:
            text_list (list of str): A list of text strings to be split into document chunks.

        Returns:
            list of Document: A list of `Document` objects, where each object contains a chunk
            of text from the input list.
    """
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))
    return documents

def add_documents_to_db(pdfs_bytes):
    """
        Extracts text from PDF files and adds the text as documents to a vector database.

        This function takes a list of PDF files in byte format, extracts text from each file,
        splits the text into smaller chunks, and adds these chunks as documents to a vector
        database for further processing and retrieval.

        Args:
            pdfs_bytes (list of bytes): A list where each element is a PDF file represented
                as bytes.

        Returns:
            None: This function does not return a value but adds the extracted documents to the vector database.
    """
    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)