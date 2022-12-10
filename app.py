import streamlit as st
import evaluate
import cohere
import numpy as np
from nltk import jaccard_distance

# create a textbox for input of response

# create a textbox for input of answer

# function to calculate metrics across these two

# test with easy stuff

# output metrics

if ('memo' not in st.session_state):
    st.session_state["Exact Match"]= 0

if ('Semantic Similarity' not in st.session_state):
    st.session_state["Semantic Similarity"]= 0






co = cohere.Client(st.secrets["cohere_key"])

st.title("Smart Flashcards! Powered by AI")

response = st.text_input(label = "Write the response to your flashcard here", value = "")
answer = st.text_input(label = "Write the correct answer to your flashcard here", value = "")

def calculate_jaccard(response, answer):
    #bug having to do with strings and unions? 

    # Given response, calculate how close it is exactly to the answer

    jd = jaccard_distance(set(answer), set(response))

    # get back similarity index which is correlate to memorization

    return 1 - jd

def calculate_ROUGE(response, answer):
    pass

def calculate_BLEU(response, answer):

    #https://huggingface.co/spaces/evaluate-metric/bleu

    bleu_score = evaluate.load("bleu")
    return bleu_score.compute(references = answer, predictions = response)


def calculate_cosine_similarity(v1, v2):

    numerator = np.dot(v1, v2)
    denom = np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))
    return numerator/denom


def calculate_semantic_similarity(response, answer):
    # cosine similarity on cohere embeddings
    embeddings = co.embed([response, answer])
    response_embeddings = embeddings.embeddings[0]
    answer_embeddings = embeddings.embeddings[1]

    cos_sim = calculate_cosine_similarity(response_embeddings, answer_embeddings)
    
    return cos_sim

def calculate_metrics(response, answer):
    #ex_match_metric = evaluate.load("exact_match")
    #ex_match_score = ex_match_metric.compute(references = [answer], prediction = [response])
    #st.session_state["Exact Match"] = ex_match_score
    #jaq = jaccard_distance(response, answer)
    semantic = calculate_semantic_similarity(response, answer)
    #bleu = calculate_BLEU(response, answer)
    st.session_state["Semantic Similarity"] = semantic



if (response != "") and (answer != ""):
    button_click = st.button("Calculate scores", on_click =calculate_metrics(response, answer))
   #st.metric(label = "Memorization", value = st.session_state["Exact Match"])
    st.metric(label = "Semantic Similarity", value = st.session_state["Semantic Similarity"])







    


