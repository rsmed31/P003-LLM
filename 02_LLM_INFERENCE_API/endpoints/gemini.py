import google.generativeai as genai
#from google.colab import userdata
import os

genai.configure(api_key="AIzaSyBOXwfitaKJVX0ov5ojRmEMRq4Pb7uk48U")
model = genai.GenerativeModel(model_name='gemini-2.5-flash')

prompt = "Hello."
response = model.generate_content(prompt)

print(response.text)