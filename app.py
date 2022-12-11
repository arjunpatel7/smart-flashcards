import streamlit as st
import evaluate
import cohere
import numpy as np
from nltk import jaccard_distance
from textacy.preprocessing.remove import punctuation
from transformers import pipeline
import pandas as pd
import requests
import json

flashcards = pd.read_csv("flashcards.csv")
# we need a way to get flashcard and iterable input...

if ('memo' not in st.session_state):
    st.session_state["Exact Match"]= 0

if ('cohere' not in st.session_state):
    st.session_state["cohere"]= 0

if ('transformers' not in st.session_state):
    st.session_state["transformers"] = 0

if ('bleu' not in st.session_state):
    st.session_state["bleu"] = 0

if ('rouge' not in st.session_state):
    st.session_state["rouge"] = 0

if ('entailment' not in st.session_state):
    st.session_state["et"] = 0

if ('card_index' not in st.session_state):
    st.session_state["card_index"] = 0

if ("current_card_question" not in st.session_state):
    st.session_state['current_card_question'] = flashcards.Question[0]
    st.session_state["current_card_answer"] = flashcards.Answer[0]


co = cohere.Client(st.secrets["cohere_key"])

#et_classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

API_URL = "https://api-inference.huggingface.co/models/roberta-large-mnli"
API_TOKEN = st.secrets["huggingface"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))



def calculate_jaccard(response, answer):
    #bug having to do with strings and unions? 

    # Given response, calculate how close it is exactly to the answer

    jd = jaccard_distance(set(answer), set(response))

    # get back similarity index which is correlate to memorization

    return 1 - jd

def calculate_ROUGE(response, answer):
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions = [response], references = [answer])["rougeL"]
    return result

def calculate_BLEU(response, answer):
    # requires length 4
    #https://huggingface.co/spaces/evaluate-metric/bleu

    bleu_score = evaluate.load("sacrebleu")
    return bleu_score.compute(references = [answer], predictions = [response],
    lowercase = True)["score"]/100


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

from sentence_transformers import SentenceTransformer, util
mod = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_ss_transformers(response, answer):
    e1 = mod.encode(response)
    e2 = mod.encode(answer)
    return util.cos_sim(e1, e2)

#def calculate_entailment(response, answer):
#    # given two sentences, determines if response follows answer or contradicts
#    candidate_labels = ["ENTAILMENT", "CONTRADICTION"]
#    result = et_classifier(response + ". " + answer,candidate_labels)
#    # return entailment score
#    return result["labels"][1]


def calculate_entailment_api(response, answer):
    data = query(
    {
        "inputs": response + ". " + answer,
    })
    data = data[0]

    #st.write(data)
    for result in data:
        # returns a list of dict that has each label and score
        if result["label"] == "ENTAILMENT":
            return result["score"]


def calculate_metrics(response, answer):
    #ex_match_metric = evaluate.load("exact_match")
    #ex_match_score = ex_match_metric.compute(references = [answer], prediction = [response])
    #st.session_state["Exact Match"] = ex_match_score
    #jaq = jaccard_distance(response, answer)
    
    response = punctuation(response)
    answer =  punctuation(answer)

    if (response == "") or (answer == ""):
        return 

    semantic_cohere = calculate_semantic_similarity(response, answer)
    semantic_transformers = calculate_ss_transformers(response, answer)
    bleu = calculate_BLEU(response, answer)
    rouge = calculate_ROUGE(response, answer)

    et = calculate_entailment_api(response, answer)

    st.session_state["et"] = et
    st.session_state["bleu"] = bleu
    st.session_state["cohere"] = semantic_cohere
    st.session_state["transformers"] = semantic_transformers
    st.session_state["rouge"] = rouge

def get_next_card():
    MAX_CARDS = len(flashcards.Question)
    if st.session_state["card_index"] + 1 != MAX_CARDS:
        st.session_state["card_index"] += 1
        st.session_state["current_card_question"] = flashcards.Question[st.session_state["card_index"]]
        st.session_state["current_card_answer"] = flashcards.Answer[st.session_state["card_index"]]
    else:
        st.session_state["card_index"] = 0


st.title("Smart Flashcards! Powered by AI")
col1, col2, col3 = st.columns(3)

with col1:
    #this will be the area someone records in
    answer = st.text_input(label = "Write your answer here", value = "")

with col2:
    # the question will be located here
    st.text_area(label = "Question", 
    value = st.session_state["current_card_question"])

with col3:
    response = st.text_area(
        label = "Correct Response", 
        value = st.session_state["current_card_answer"])


calc_button_click = st.button("Calculate scores", on_click =calculate_metrics(response, answer))
next_card = st.button("Next Card", on_click = get_next_card())

if (response != ""):
   #st.metric(label = "Memorization", value = st.session_state["Exact Match"])
    st.metric(label = "BLEU", value = st.session_state["bleu"])
    st.metric(label = "ROUGE", value = st.session_state["rouge"])
    st.metric(label = "Entailment Probability", value = st.session_state["et"])
    st.metric(label = "Semantic Similarity Cohere", value = st.session_state["cohere"])
    st.metric(label = "Semantic Similarity Transformers", value = st.session_state["transformers"])







    


