from flask import Flask, render_template, request, jsonify
from transformers import TextClassificationPipeline, TFAutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
import tensorflow as tf
import random

app = Flask(__name__, template_folder='home')
 
# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_load_path = "diagnosisModel"
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(model_load_path)

# Create a text classification pipeline
pipe = TextClassificationPipeline(model=loaded_model, tokenizer=tokenizer, top_k=33)

# Translator function
# def translator(text):
#     to_translate = text
#     translated = GoogleTranslator(source='auto', target='en').translate(to_translate)
#     return translated

# def translatorFa(text):
#     to_translate = text
#     translated = GoogleTranslator(source='auto', target='fa').translate(to_translate)
#     return translated

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_message = request.json["text"]
    # user_message_translated = translator(user_message)
    bot_response = pipe(user_message)
    bot_response_text = bot_response[0][:1]
    bot_response_text = bot_response_text[0]['label']
    bot_response_text = f'You may have {bot_response_text}. It is better to see a doctor as soon as possible. Do you want me to suggest the medicine you need?'
    # bot_response_text = translatorFa(bot_response_text)
    return jsonify({"response": bot_response_text})

if __name__ == "__main__":
    app.run(debug=True)