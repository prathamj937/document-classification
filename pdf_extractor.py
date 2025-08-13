import os
import streamlit as st
import pdfplumber
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image
from groq import Groq
import io
import base64

# Initialize Groq client
client = Groq(api_key="gsk_Ri6OY46IxYHEgW7cEFL8WGdyb3FYTCDTqTJw28YGfE9UnEY8sFd")

# Load models
clf = pickle.load(open("logreg_model.pkl", "rb"))
sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Function to extract text using pdfplumber + Groq OCR fallback
def extract_text_with_groq(pdf_file):
    images_content = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=300).original
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            images_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    
    # Send all images in one Groq request
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "Extract all text from these PDF pages in correct order."}] + images_content
        }],
        temperature=0
    )
    return resp.choices[0].message.content


st.title("📄 Document Classifier (Groq OCR)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        doc_text = extract_text_with_groq(uploaded_file)
    
    if doc_text:
        with st.spinner("Generating prediction..."):
            embedding = sbert_model.encode([doc_text])
            prediction = clf.predict(embedding)[0]
        
        st.success(f"**Predicted Class:** {prediction}")
        st.text_area("Extracted Text", doc_text, height=200)
    else:
        st.error("No text found in PDF.")
