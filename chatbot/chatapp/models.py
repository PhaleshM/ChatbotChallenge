from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    embedding = models.TextField(blank=True)
    document = models.FileField(upload_to='chatapp/documents/')
    content_type = models.CharField(max_length=50)