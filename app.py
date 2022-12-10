import streamlit as st
import evaluate
import cohere
import numpy as np
from nltk import jaccard_distance
from textacy.preprocessing.remove import punctuation
from transformers import pipeline

# create a textbox for input of response

# create a textbox for input of answer

# function to calculate metrics across these two

# test with easy stuff

# output metrics

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


et_classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')


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

def calculate_entailment(response, answer):
    # given two sentences, determines if response follows answer or contradicts
    candidate_labels = ["ENTAILMENT", "CONTRADICTION"]
    result = et_classifier(response + ". " + answer,candidate_labels)
    # return contradiction score
    return result["labels"][0]

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

    et = calculate_entailment(response, answer)

    st.session_state["et"] = et
    st.session_state["bleu"] = bleu
    st.session_state["cohere"] = semantic_cohere
    st.session_state["transformers"] = semantic_transformers
    st.session_state["rouge"] = rouge



if (response != "") and (answer != ""):
    button_click = st.button("Calculate scores", on_click =calculate_metrics(response, answer))
   #st.metric(label = "Memorization", value = st.session_state["Exact Match"])
    st.metric(label = "BLEU", value = st.session_state["bleu"])
    st.metric(label = "ROUGE", value = st.session_state["rouge"])
    st.metric(label = "Contradiction Probability?", value = st.session_state["et"])
    st.metric(label = "Semantic Similarity Cohere", value = st.session_state["cohere"])
    st.metric(label = "Semantic Similarity Transformers", value = st.session_state["transformers"])







    


