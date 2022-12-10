import streamlit as st

# create a textbox for input of response

# create a textbox for input of answer

# function to calculate metrics across these two

# test with easy stuff

# output metrics

st.title("Smart Flashcards! Powered by AI")

response = st.text_input(value = "Write the response to your flashcard here")
answer = st.text_input(value = "Write the correct answer to your flashcard here")


def calculate_memorization(response, answer):
    # Given response, calculate how close it is exactly to the answer
    pass

def calculate_ROUGE(response, answer):
    pass

def calculate_BLEU(response, answer):
    pass

def calculate_semantic_similarity(response, answer):
    # cosine similarity on cohere embeddings
    pass





    


