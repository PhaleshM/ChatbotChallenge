from django.shortcuts import render, redirect,get_object_or_404
from django.http import HttpResponse
from .forms import DocumentForm
from .models import Document
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
load_dotenv()

# gemini_api_key = os.environ["GEMINI_API_KEY"]
google_api_key =os.environ["GEMINI_API_KEY"]
# genai.configure(api_key = gemini_api_key)
huggingfacehub_api_token=os.environ['huggingfacehub_api_token']

# embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from django.http import JsonResponse

def chat_page(request, document_id=None):
    document = None
    if document_id:
        document = get_object_or_404(Document, pk=document_id)
    else:
        return HttpResponse("URL not Found")
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        user_question = request.POST.get('user_question', '')
        chatbot_response,history = get_chatbot_response(user_question, document_id)
        return JsonResponse({'user_question': user_question, 'chatbot_response': chatbot_response})

    return render(request, 'chatapp/chat_page.html', {'document': str(document.title)})



# facebook/bart-large-cnn
# llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.5, "max_length": 800},
#                     huggingfacehub_api_token=huggingfacehub_api_token)
# llm= OpenAI(model_name="gpt-3.5-turbo-instruct")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
document_variable_name = "context"
prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)

# Now, use the pre-created chains in the function
def get_chatbot_response(user_question, document_id, chat_history=None):
    path = 'chatapp/vector_db/' + str(document_id)
    load_db = FAISS.load_local(folder_path=path, embeddings=embedding_model)
    retriever = load_db.as_retriever(search_kwargs={"k": 1})

    if chat_history is None:
        chat_history = []  # Store the chat history here

    template = (
        "Combine the chat history and follow up question into "
        "a standalone question. Give answer in 400 characters. Chat History: {chat_history}"
        "Follow up question: {question}"
    )
    prompt = PromptTemplate.from_template(template)

    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever,
        question_generator=question_generator_chain,
    )
    # chat_history.append({'role': 'user', 'content': user_question})

    # Retrieve the response
    response = chain({
        'question': user_question,
        'chat_history': chat_history
    })
    bot_response = response.get('answer', '')
    chat_history.append({'role': 'bot', 'content': bot_response})
    return bot_response, chat_history

def process_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            # Determine content type dynamically
            content_type = request.FILES['document'].content_type
            document.content_type = content_type

            document.save()
            # Handle different document types
            if content_type == 'application/pdf':
                process_pdf(request.FILES['document'],document.id)
            else:
                return HttpResponse("Unsupported document type")

            return redirect('chat_page',document_id=document.id)

    else:
        form = DocumentForm()

    return render(request, 'chatapp/document_form.html', {'form': form})


def process_pdf(file,id):
    # Extract text from PDF and generate embedding
    document = extract_text_from_pdf(file)
    return generate_embedding(document,id)


from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(file):
    # Extract text from PDF using PyMuPDF from an in-memory file
    directory="chatapp/documents/"+str(file)
    file_loader=PyPDFLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

def generate_embedding(text,id):
    # Generate embedding using Sentence Transformers
    documents=chunk_data(text)
    vector_db = FAISS.from_documents(documents=documents,embedding=embedding_model)
    path='chatapp/vector_db/'+str(id)
    vector_db.save_local(path)

    
