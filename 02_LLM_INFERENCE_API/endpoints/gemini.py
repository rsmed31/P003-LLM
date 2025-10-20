import google.generativeai as genai



def configureGemini(apiKey):
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel(model_name='gemini-2.5-flash')
    return model



def callGemini(model, prompt):
    return model.generate_content(prompt)

model = configureGemini("AIzaSyBOXwfitaKJVX0ov5ojRmEMRq4Pb7uk48U")
response = callGemini(model, 'What is the capital of Egypt?')

print(response.text)