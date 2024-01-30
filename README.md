"""
ChatBot Django Project

## Overview

This Django project implements a ChatBot system with conversational retrieval capabilities. Users can upload documents, and the ChatBot will process them to generate responses to user queries.

## Features

1. **Document Processing:**
   - Users can upload documents in various formats, including PDFs.
   - The system extracts text from PDFs using PyMuPDF and generates embeddings using Sentence Transformers.

2. **Conversational Retrieval:**
   - The ChatBot employs conversational retrieval to provide context-aware responses.
   - It uses a combination of document embeddings and a language model (ChatGoogleGenerativeAI) for response generation.

3. **Chat Interface:**
   - The project includes a chat interface where users can interact with the ChatBot.
   - The conversation history is stored and utilized for context-aware responses.

4. **Django Structure:**
   - Django's URL configuration routes requests to the appropriate views.
   - Views handle document processing, chat page rendering, and AJAX-based user interactions.

5. **Backend Components:**
   - Utilizes FAISS for vector storage and retrieval.
   - Embeddings are generated using SentenceTransformerEmbeddings.
   - Chains like StuffDocumentsChain and ConversationalRetrievalChain orchestrate the processing flow.

6. **Configuration:**
   - Project settings are configured through environment variables, such as GEMINI_API_KEY and huggingfacehub_api_token.
   - Google API key is used for the ChatGoogleGenerativeAI model.

## Setup and Installation

1. Clone the repository: `git clone <repository_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file.

## Usage

1. Run the Django development server: `python manage.py runserver`
2. Access the application at `http://localhost:8000/`

## Dependencies

- Django: Web framework for building the application.
- langchain-community: Library for document processing, embeddings, and chains.
- Sentence Transformers: Library for generating embeddings from sentences.
- PyMuPDF: Library for extracting text from PDFs.
