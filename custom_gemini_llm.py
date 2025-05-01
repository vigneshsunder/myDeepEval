import requests
from deepeval.models import DeepEvalBaseLLM


class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyA1Dtr8Q4U4xUpbXqnwPPOHKhIGDG7MsIg"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        response_json = response.json()
        llm_response = response_json['candidates'][0]['content']['parts'][0]['text']
        if 'error' in response_json:
            raise Exception(f"Error from Gemini API: {response_json['error']['message']}")
        return llm_response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Gemini 2.0 Flash"