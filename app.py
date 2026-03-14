import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(text, max_len):

    inputs = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary



st.title("AI Text Summarization App")

st.write("Paste a long article and generate a short summary.")

text_input = st.text_area("Enter Text")

max_len = st.slider("Select Summary Length", 30, 120, 60)

if st.button("Generate Summary"):

    if text_input:

        summary = generate_summary(text_input, max_len)

        st.subheader("Summary")
        st.write(summary)

        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

        original_word_count = len(text_input.split())
        summary_word_count = len(summary.split())

        st.write("Original Text Word Count:", original_word_count)
        st.write("Summary Word Count:", summary_word_count)

        compression_ratio = round(summary_word_count / original_word_count, 2)

        st.write("Compression Ratio:", compression_ratio)

    else:
        st.write("Please enter some text.")