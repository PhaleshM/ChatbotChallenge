from django.urls import path
from .views import process_document,chat_page

app_name = 'chatbot'

urlpatterns = [
    path('chat_page/<int:document_id>/', chat_page, name='chat_page'),
    path('process_document/', process_document, name='process_document'),
    # Add other URL patterns as needed
]
