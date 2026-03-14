from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_summary(text):

    inputs = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=60,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


if __name__ == "__main__":

    article = """
    Artificial Intelligence is transforming industries by automating tasks,
    improving decision making, and enabling machines to learn from data.
    Companies across healthcare, finance, and technology are investing heavily
    in AI systems to improve efficiency and create innovative products.
    """

    summary = generate_summary(article)

    print("\nOriginal Text:\n", article)
    print("\nGenerated Summary:\n", summary)