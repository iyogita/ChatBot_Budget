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
    from collections import deque

    class UserContext:
        def __init__(self, max_size=5):
            self.context = deque(maxlen=max_size)

        def add_interaction(self, user_input, bot_response):
            self.context.append({'user_input': user_input, 'bot_response': bot_response})

        def get_context(self):
            return list(self.context)
    ```

8. **Refine Query with GPT-2:**
    ```python
    from transformers import pipeline, set_seed

    generaUsage
Upload a PDF File:

Use the google.colab library to upload the PDF file to the Colab environment.
Extract Text from PDF:

Use pdfplumber to extract text from the uploaded PDF file.
Segment Text:

Split the extracted text into manageable chunks for processing.
Generate Embeddings:

Utilize the sentence-transformers library to generate embeddings for the text chunks.
Build FAISS Index:

Create a FAISS index to store and manage the embeddings for efficient similarity search.
Generate Query Embeddings:

Convert user queries into embeddings for comparison against the stored text chunk embeddings.
Search FAISS Index:

Perform a search on the FAISS index to find the most relevant text chunks based on the query embedding.
Maintain User Context:

Keep track of the last 5 user interactions to provide better query refinement.
Refine User Queries with GPT-2:

Use GPT-2 to refine user queries based on the maintained context for more accurate results.
Retrieve and Present Relevant Text Chunks:

Retrieve and display the most relevant text chunks to the user.
Performance Evaluation
Text Extraction Performance:

Measure the time taken to extract text from PDF files.
Text Segmentation Performance:

Measure the time taken to segment the extracted text into chunks.
Embedding Generation Performance:

Measure the time taken to generate embeddings for the text chunks.
FAISS Index Building Performance:

Measure the time taken to build the FAISS index.
Query Response Performance:

Measure the time taken to generate query embeddings and retrieve results from the FAISS index.
Overall System Performance:

Measure the total processing time for the entire workflow from text extraction to query response.
Documentation
System Architecture: Include a high-level architecture diagram and description of the system components.
Implementation Details: Provide details on the implementation of each component and how they interact.
Deployment Instructions: Provide instructions for setting up and deploying the system.
Execution Instructions: Provide a step-by-step guide for running the system.
Performance Evaluation Results: Include a summary of the performance evaluation results.
Deliverables
High-Level Architecture Diagram
Source Code
README with deployment instructions and performance evaluation results
Video demonstrating the running application
Contact
For any questions or support, please contact [Your Name] at [Your Email].
        
       
