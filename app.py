import streamlit as st
import cohere
import numpy as np
from textacy.preprocessing.remove import punctuation
from transformers import pipeline
import pandas as pd
import requests
import json

flashcards = pd.read_csv("flashcards.csv")
# we need a way to get flashcard and iterable input...

if ('grade' not in st.session_state):
    st.session_state["grade"] = ""
if ('cohere' not in st.session_state):
    st.session_state["cohere"]= 0

if ('entailment' not in st.session_state):
    st.session_state["et"] = 0

if ('num_calc' not in st.session_state):
    st.session_state["num_calc"] = 0

if ('card_index' not in st.session_state):
    st.session_state["card_index"] = 0

if ("current_card_question" not in st.session_state) and ("current_card_answer" not in st.session_state):
    st.session_state["current_card_question"] = flashcards.Question[0]
    st.session_state["current_card_answer"] = flashcards.Answer[0]

co = cohere.Client(st.secrets["cohere_key"])

API_URL = "https://api-inference.huggingface.co/models/roberta-large-mnli"
API_TOKEN = st.secrets["huggingface"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


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


def calculate_entailment_api(response, answer):
    data = query(
    {
        "inputs": response + ". " + answer,
    })
    data = data[0]

    #st.write(data)
    for result in data:
        # returns a list of dict that has each label and score
        if result["label"] == "CONTRADICTION":
            return result["score"]


def calculate_metrics(resp, ans):
    
    response = punctuation(resp)
    answer =  punctuation(ans)


    semantic_cohere = calculate_semantic_similarity(response, answer)
    et = calculate_entailment_api(response, answer)

    st.session_state["et"] = et
    st.session_state["cohere"] = semantic_cohere


st.title("SmartFlash Companion: Powered by AI!")

#dropdown for flashcards -> medical terminology deck

with st.expander("Preview Medical Terminology Deck here!"):
    st.dataframe(flashcards)

question = st.text_area(label = "Question", 
    value = st.session_state["current_card_question"])

response = st.text_area(label = "Write your answer here and wait a few seconds for feedback...", value = "", key = "response")


if st.session_state.num_calc > 0:
    st.write("The number of calculations is")
    st.write(st.session_state.num_calc)
    del st.session_state.response


def get_next_card():
    # clear the original values
    st.session_state["response"] = ""
    MAX_CARDS = len(flashcards.Question)
    if st.session_state["card_index"] + 1 < MAX_CARDS:
        st.session_state["card_index"] += 1
        st.session_state["current_card_question"] = flashcards.Question[st.session_state["card_index"]]
        st.session_state["current_card_answer"] = flashcards.Answer[st.session_state["card_index"]]
    else:
        st.session_state["card_index"] = 0
        st.session_state["current_card_question"] = flashcards.Question[st.session_state["card_index"]]
        st.session_state["current_card_answer"] = flashcards.Answer[st.session_state["card_index"]]

next_card = st.button("Next Card", on_click = get_next_card)

def clear_entry():
    st.session_state["response"] = ""

clear_card = st.button("Clear Entry", on_click=clear_entry)

if (response != ""):
    calculate_metrics(response, st.session_state.current_card_answer)
    # st.metric(label = "Memorization", value = st.session_state["Exact Match"])
    # st.metric(label = "BLEU", value = st.session_state["bleu"])
    # st.metric(label = "ROUGE", value = st.session_state["rouge"])
    # st.metric(label = "Entailment Probability", value = 1 - st.session_state["et"])
    # st.metric(label = "Semantic Similarity Cohere", value = st.session_state["cohere"])
    # st.metric(label = "Semantic Similarity Transformers", value = st.session_state["transformers"])   
    cohere_score = st.session_state.cohere
    et_score = 1 - st.session_state.et

    # using entailment score inspired by discussion here
    # https://stackoverflow.com/questions/69374258/sentence-similarity-models-not-capturing-opposite-sentences

    total_evaluation = (cohere_score * 0.45) + (et_score * 0.55)
    # st.metric(label = "Correctness Score", value = total_evaluation)
    if total_evaluation >= 0.80:
        # mark as correct, get next card, reset correctness
        st.success("You got that correct!")
        st.session_state.grade = 1
        st.text_area(label = "The actual answer is...", value = st.session_state["current_card_answer"])
    elif total_evaluation >= 0.70:
        st.session_state.grade = 0
        st.success("You are so close, keep trying!")
    else:
        st.error("Try again please! Our custom score shows you are off the mark")


