from transformers import pipeline

summarizer = pipeline("summarization")

def generate_summary(text):

    summary = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return summary[0]['summary_text']


if __name__ == "__main__":

    article = """
    Artificial Intelligence is transforming industries by automating tasks,
    improving decision making, and enabling machines to learn from data.
    Companies across healthcare, finance, and technology are investing heavily
    in AI systems to improve efficiency and create innovative products.
    """

    result = generate_summary(article)

    print("\nOriginal Text:\n", article)
    print("\nGenerated Summary:\n", result)