# QA Bot for PDF Files

## Overview

This project is a QA bot that processes PDF files to extract text, generate embeddings, and store them in a vector database (FAISS). Users can query the system, which retrieves relevant text chunks based on similarity scores. The system integrates with a GPT model for refining user queries and improving search accuracy.

## Features

- Extract text from PDFs
- Segment text into manageable chunks
- Generate embeddings for text chunks
- Store and manage embeddings in a FAISS index
- Generate embeddings for user queries
- Search the FAISS index to find relevant text chunks
- Maintain a context of the last 5 user interactions
- Refine user queries using GPT-2
- Retrieve and present the most relevant text chunks to users

## Requirements

- Python 3.x
- `pdfplumber`
- `sentence-transformers`
- `faiss-cpu`
- `transformers`
- `google.colab`

## Installation

1. Clone the Repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required libraries:
    ```bash
    pip install pdfplumber
    pip install sentence-transformers
    pip install faiss-cpu
    pip install transformers
    ```

## Usage

1. **Upload PDF File:**
    ```python
    Use the google.colab library to upload the PDF file to the Colab environment.
    ```

2. **Extract Text from PDF:**
    ```python
    Use pdfplumber to extract text from the uploaded PDF file.
    ```

3. **Segment Text:**
    ```python
    Split the extracted text into manageable chunks for processing.    
    ```

4. **Generate Embeddings:**
    ```python
    Utilize the sentence-transformers library to generate embeddings for the text chunks.
    ```

5. **Build FAISS Index:**
    ```python
    Create a FAISS index to store and manage the embeddings for efficient similarity search.
    ```

6. **Search FAISS Index:**
    ```python
    Convert user queries into embeddings for comparison against the stored text chunk embeddings.
    ```

7. **Manage User Context:**
    ```python
Keep track of the last 5 user interactions to provide better query refinement.
    ```

8. **Refine Query with GPT-2:**
    ```python
    Use GPT-2 to refine user queries based on the maintained context for more accurate results.
     ```

9. **Retrieve and Present Relevant Text Chunks:**
    ```python
    Retrieve and display the most relevant text chunks to the user.    
    ```

## Performance Evaluation

4. **Text Extraction Performance:**
    ```python
    Measure the time taken to extract text from PDF files.
    ```

5. **Text Segmentation Performance:**
    ```python
    Measure the time taken to segment the extracted text into chunks.
    ```

6. **Embedding Generation Performance:**
    ```python
   Measure the time taken to build the FAISS index.
    ```

7. **Query Response Performance:**
    ```python
Measure the time taken to generate query embeddings and retrieve results from the FAISS index.
    ```

8. **Overall System Performance:**
    ```python
    Measure the total processing time for the entire workflow from text extraction to query response.
     ```
       
