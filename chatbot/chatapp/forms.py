from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from .models import Document
from django.utils.translation import gettext_lazy as _

class FileSizeValidator:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, value):
        if value.size > self.max_size:
            raise ValidationError(_('File size must be no more than %(max_size)s MB.'), params={'max_size': self.max_size / (1024 * 1024)})


class DocumentForm(forms.ModelForm):
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB in this example

    document = forms.FileField(
        validators=[
            FileExtensionValidator(allowed_extensions=['pdf', 'docx', 'txt']),  # Add allowed file extensions if needed
            FileSizeValidator(max_size=MAX_SIZE)
        ]
    )

    class Meta:
        model = Document
        fields = ['title', 'document']
