import google.generativeai as genai
import os
import json
import requests

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'inference.json')
with open(config_path, 'r') as f:
    CONFIG = json.load(f)

def configureGemini(apiKey, model_name):
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel(model_name=model_name)
    return model

def callGemini(model, prompt):
    response = model.generate_content(prompt)
    return response.text

def callLlama(api_link, prompt, api_key=None):
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    payload = {
        "prompt": prompt
    }
    
    response = requests.post(api_link, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get('text', response.text)

def generate(model_name, prompt):
    """
    Main inference function that routes to appropriate API based on model.
    
    Args:
        model_name: "gemini" or any other name for local model
        prompt: The input prompt string
    
    Returns:
        Generated text response
    """
    if model_name == "gemini":
        config = CONFIG['gemini']
        model = configureGemini(config['api_key'], config['model'])
        return callGemini(model, prompt)
    else:
        # Treat any non-gemini model as local
        config = CONFIG.get(model_name, CONFIG['llama'])
        return callLlama(config['api_link'], prompt, config.get('api_key'))

# Example usage
if __name__ == "__main__":
    model_name = "gemini"  # Change to "llama" to use local API
    prompt = 'What is the capital of Egypt?'
    
    try:
        response = generate(model_name, prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")