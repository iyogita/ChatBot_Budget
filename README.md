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

1. Install the required libraries:
    ```bash
    pip install pdfplumber
    pip install sentence-transformers
    pip install faiss-cpu
    pip install transformers
    ```

## Usage

1. **Upload PDF File:**
    ```python
    from google.colab import files
    uploaded = files.upload()
    pdf_path = list(uploaded.keys())[0]
    ```

2. **Extract Text from PDF:**
    ```python
    import pdfplumber

    def extract_text_from_pdf(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    ```

3. **Segment Text:**
    ```python
    def segment_text(text, chunk_size=512):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    ```

4. **Generate Embeddings:**
    ```python
    from sentence_transformers import SentenceTransformer

    def generate_embeddings(text_chunks):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text_chunks, convert_to_tensor=True, normalize_embeddings=True)
    ```

5. **Build FAISS Index:**
    ```python
    import faiss
    import numpy as np

    def build_faiss_index(embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return index
    ```

6. **Search FAISS Index:**
    ```python
    def search_index(index, query_embedding, k=5):
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        D, I = index.search(query_embedding, k)
        return I
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

    generator = pipeline('text-generation', model='gpt2')

    def refine_query_with_gpt(user_context, query):
        set_seed(42)
        context = user_context.get_context()
        context_text = "\n".join([f"User: {interaction['user_input']}\nBot: {interaction['bot_response']}" for interaction in context])
        prompt = f"{context_text}\nUser: {query}\nBot:"
        response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        refined_response = response[len(prompt):].strip()
        return refined_response
    ```

9. **Performance Evaluation:**
    ```python
    import time

    def performance_evaluation(pdf_path, query_text):
        # Text Extraction Performance
        start_time = time.time()
        pdf_text = extract_text_from_pdf(pdf_path)
        end_time = time.time()
        extraction_time = end_time - start_time

        # Text Segmentation Performance
        start_time = time.time()
        text_chunks = segment_text(pdf_text)
        end_time = time.time()
        segmentation_time = end_time - start_time
        chunk_count = len(text_chunks)

        # Embedding Generation Performance
        start_time = time.time()
        embeddings = generate_embeddings(text_chunks)
        end_time = time.time()
        embedding_generation_time = end_time - start_time
        embedding_size = embeddings.shape[1]

        # FAISS Index Building Performance
        start_time = time.time()
        index = build_faiss_index(embeddings)
        end_time = time.time()
        index_building_time = end_time - start_time
        index_size = index.ntotal

        # Query Response Performance
        query_embedding = generate_embeddings([query_text])[0]
        start_time = time.time()
        results = search_index(index, query_embedding)
        end_time = time.time()
        query_time = end_time - start_time

        # Overall System Performance
        total_start_time = time.time()
        # Complete workflow
        pdf_text = extract_text_from_pdf(pdf_path)
        text_chunks = segment_text(pdf_text)
        embeddings = generate_embeddings(text_chunks)
        index = build_faiss_index(embeddings)
        query_embedding = generate_embeddings([query_text])[0]
        results = search_index(index, query_embedding)
        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        # Report
        print(f"Text Extraction Time: {extraction_time} seconds")
        print(f"Text Segmentation Time: {segmentation_time} seconds")
        print(f"Number of Chunks: {chunk_count}")
        print(f"Embedding Generation Time: {embedding_generation_time} seconds")
        print(f"Embedding Size: {embedding_size}")
        print(f"Index Building Time: {index_building_time} seconds")
        print(f"Index Size: {index_size}")
        print(f"Query Response Time: {query_time} seconds")
        print(f"Total Processing Time: {total_processing_time} seconds")
    ```

10. **Context Preservation Test:**
    ```python
    def test_context_preservation():
        user_context = UserContext(max_size=5)
        
        # Simulate adding more than 5 interactions
        for i in range(1, 7):
            user_context.add_interaction(f"Question {i}", f"Answer {i}")
        
       
