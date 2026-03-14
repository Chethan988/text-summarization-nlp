# NLP Text Summarization using Transformers

## Overview
This project implements an AI-powered text summarization system that generates concise summaries from long textual documents using transformer-based deep learning models.

## Features
- Automatic text summarization
- Transformer-based NLP model
- Interactive web application
- User-friendly interface

## Technologies Used
- Python
- Transformers
- PyTorch
- Streamlit
- Natural Language Processing (NLP)

## Model
The system uses the pretrained model:
facebook/bart-large-cnn

This model is designed for abstractive text summarization and is capable of generating human-like summaries.

## Project Structure
text-summarization-nlp
│
├── data
├── notebooks
├── src
│   └── summarizer.py
│
├── app.py
├── requirements.txt
└── README.md
## How to Run

1 Install dependencies
pip install -r requirements.txt

2 Run the Streamlit app
streamlit run app.py