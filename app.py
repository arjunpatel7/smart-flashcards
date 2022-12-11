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

if ('grade' not in st.session_state):
    st.session_state["grade"] = ""
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

if ("current_card_question" not in st.session_state) and ("current_card_answer" not in st.session_state):
    st.session_state["current_card_question"] = flashcards.Question[0]
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


def get_next_card():
    MAX_CARDS = len(flashcards.Question)
    if st.session_state["card_index"] + 1 < MAX_CARDS:
        st.session_state["card_index"] += 1
        st.session_state["current_card_question"] = flashcards.Question[st.session_state["card_index"]]
        st.session_state["current_card_answer"] = flashcards.Answer[st.session_state["card_index"]]
    else:
        st.session_state["card_index"] = 0

def calculate_metrics(resp, ans):
    #ex_match_metric = evaluate.load("exact_match")
    #ex_match_score = ex_match_metric.compute(references = [answer], prediction = [response])
    #st.session_state["Exact Match"] = ex_match_score
    #jaq = jaccard_distance(response, answer)
    
    response = punctuation(resp)
    answer =  punctuation(ans)


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


st.title("Smart Flashcards! Powered by AI")



st.text_area(label = "Question", 
    value = st.session_state["current_card_question"])

response = st.text_area(label = "Write your answer here", value = "")
next_card = st.button("Next Card")
if next_card:
    get_next_card()


#answer = st.text_area(
#        label = "Answer", 
#        value = st.session_state["current_card_answer"])


#calc_button_click = st.button("Calculate scores", on_click =calculate_metrics,
#args = (response, answer))
#do st. for grade, change it, and have a routine on the chacne to update card


if (response != ""):
    calculate_metrics(response, st.session_state.current_card_answer)
   #st.metric(label = "Memorization", value = st.session_state["Exact Match"])
    st.metric(label = "BLEU", value = st.session_state["bleu"])
    st.metric(label = "ROUGE", value = st.session_state["rouge"])
    st.metric(label = "Entailment Probability", value = st.session_state["et"])
    st.metric(label = "Semantic Similarity Cohere", value = st.session_state["cohere"])
    st.metric(label = "Semantic Similarity Transformers", value = st.session_state["transformers"])   
    if st.session_state.rouge >= 0.7:
        # mark as correct, get next card, reset correctness
        st.success("You got that correct!")
        st.session_state.grade = 1
    else:
        st.session_state.grade = 0


    


