import gradio as gr
import scipy.special
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax  # Corrected import statement
import torch

# ... (rest of your code)

#setup
model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# preprocess text


# Sentiment analysis function using the trained model
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move input tensor to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Get the probability distribution over the classes
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence_percentage = probabilities[0, predicted_class].item() * 100

    # Map the predicted class index to a human-readable label
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment = sentiment_labels[predicted_class]
    return sentiment, confidence_percentage

# Gradio interface setup
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your tweet here..."),
    outputs=[gr.Label("Sentiment Analysis Result"), gr.Label("Confidence Percentage")],
    examples=[["This is wonderful!"]],
    theme="compact",
    title="Twitter Sentiment Analysis"
)

# Launch the Gradio interface with debugging enabled
demo.launch(share=True, debug=True)
