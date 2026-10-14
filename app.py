import streamlit as st
from transformers import pipeline


st.title("Amna Shahzad - B04-0923-000008 - Sentiment Analysis App")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


classifier = load_model()


st.write("Enter three texts and analyze their sentiment")


text1 = st.text_area("Sentence 1")
text2 = st.text_area("Sentence 2")
text3 = st.text_area("Sentence 3")


if st.button("Analyze Sentiment"):

    inputs = [text1, text2, text3]

    inputs = [text for text in inputs if text.strip()]


    if len(inputs) == 0:
        st.warning("Please enter text")

    else:

        results = classifier(inputs)

        for i, result in enumerate(results):

            st.write("Input:", inputs[i])

            st.write("Sentiment:", result["label"])

            st.write(
                "Confidence:",
                round(result["score"] * 100, 2),
                "%"
            )

            st.divider()
