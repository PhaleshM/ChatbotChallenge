from transformers import pipeline

class LanguageModel:
    def __init__(self, model_name_or_path='gpt2'):
        self.model = pipeline('text-generation', model=model_name_or_path)

    def generate_response(self, prompt, max_length=50):
        return self.model(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
