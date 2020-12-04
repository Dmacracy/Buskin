from entities import *
import time, os, sys, re, json, collections
from multiprocessing import Pool

from fuzzywuzzy import fuzz
import pandas as pd
# Load your usual SpaCy model (one of SpaCy English models)
import spacy
nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp, blacklist=True)


from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as nnf
tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original", return_dict=True).to('cuda')
model.eval()

import numpy as np

emotions = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',\
            'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',\
            'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',\
            'remorse','sadness','surprise','neutral']
reduced_emotions = { 'admiration' : 'pos', 'amusement' : 'pos', 'anger' : 'neg', 'annoyance' : 'neg', 'approval' : 'pos',
            'caring' : 'pos', 'confusion' : 'amb', 'curiosity' : 'amb', 'desire' : 'pos', 'disappointment' : 'neg', 'disapproval' : 'neg',
            'disgust' : 'neg', 'embarrassment' : 'neg', 'excitement' : 'pos', 'fear' : 'neg', 'gratitude' : 'pos', 'grief' : 'neg',
            'joy' : 'pos', 'love' : 'pos', 'nervousness' : 'neg', 'optimism' : 'pos','pride' : 'pos', 'realization' : 'amb',
            'relief' : 'pos', 'remorse' : 'neg', 'sadness' : 'neg', 'surprise' : 'amb', 'neutral' : 'amb'}


def parse_into_sents_corefs(inp):
    text, par_id = inp
    doc = nlp(text)
    # parse into sentences
    sentences = []
    sentence_id_for_tokens = []
    for s, sent in enumerate(doc.sents):
        tokens = doc[sent.start:sent.end]
        sentence_id_for_tokens += [s] * len(tokens)
        token_tags = [TOKEN_TAGS(i, token.text, token.lemma_, token.pos_, token.pos_, token.dep_) for i, token in enumerate(tokens)]
        emotion_tags = Emotion(None,None,None)
        sentences.append(Sentence(s, par_id, sent.text, token_tags, emotion_tags))
    
    if doc._.has_coref:
        corefs = {}
        for cluster in doc._.coref_clusters:
            if cluster.main.text.lower() in corefs:
                corefs[cluster.main.text.lower()]+=[(mention.text.lower(),sentence_id_for_tokens[mention.start]) for mention in cluster.mentions]
            else:
                corefs[cluster.main.text.lower()]=[(mention.text.lower(),sentence_id_for_tokens[mention.start]) for mention in cluster.mentions]
    return sentences, corefs



def get_merged_corefs(coref_dicts, max_fuzz = 70):
    main_coref = {}
    for dict_ in coref_dicts:
        for k,v in dict_.items():
            if k in main_coref:
                main_coref[k]+=v
            else:
                main_coref[k] = v
    
    merged_coref = {}
    for k,v in main_coref.items():
        added = 0
        for merged_char in merged_coref.keys():
            if fuzz.partial_ratio(merged_char, k) > max_fuzz:
                merged_coref[merged_char] += v
                added = 1
                break
        if added==0:
            merged_coref[k] = v

    return merged_coref

def convert_text_to_chunks(text):
    # split on newlines followed by space
    pars = re.split('\n\s', text)   
    # Replace newline chars
    pars = [par.replace("\n", " ") for par in pars]
    # Remove empty pars
    pars = [par for par in pars if len(par) > 0]
    
    #Preprocess "paragraphs" that are actually quotes or single lined text
    final_pars = []
    for p,paragraph in enumerate(pars):
        
        if paragraph.count(".")<5:
            if p==0:
                final_pars.append(paragraph)
            else:
                final_pars[-1] = final_pars[-1] + " " + paragraph
        else:
            final_pars.append(paragraph)
    
    TOTAL_CHUNKS = 5
    final_chunks = []
    chunk_id = 0
    pars_per_chunk = round(len(final_pars)/TOTAL_CHUNKS) 
    while chunk_id * pars_per_chunk < len(final_pars):
        final_chunks.append((' '.join(final_pars[chunk_id * pars_per_chunk : min((chunk_id + 1 ) * pars_per_chunk,len(final_pars))]), chunk_id))
        chunk_id+=1
    return final_chunks

def get_emotion_per_batch(batch):
    inputs = tokenizer(batch, is_split_into_words=True, return_tensors='pt', padding=True).to('cuda')
    outputs = model(**inputs)
    logits = outputs.logits
    probs = nnf.softmax(logits, dim=1).cpu().data.numpy()
    emotion_res = [emotions[x] for x in np.argmax(probs, axis=1)]
    emo_prob = list(np.max(probs, axis=1))
    mini_emotion_res = [reduced_emotions[emotion] for emotion in emotion_res]
    return  (emotion_res, mini_emotion_res, emo_prob)

def generate_sentence_batches(sentences, BATCH_SIZE=16):
    i=0
    while i*BATCH_SIZE<len(sentences):
        subset = sentences[i*BATCH_SIZE: min((i+1)*BATCH_SIZE, len(sentences))]
        subset = [[t.token for t in s.token_tags] for s in subset]
        i+=1
        yield subset

def merge_emotions_to_sentences(sentences, emotion_batches):
    emotions = []
    mini_emotions = []
    probs = []
    for e,m,p in emotion_batches:
        emotions+=e
        mini_emotions+=m
        probs+=p

    assert len(emotions) == len(mini_emotions) 
    assert len(emotions) == len(probs) 
    assert len(emotions) == len(sentences)

    for i in range(len(sentences)):
        sentences[i].emotion_tags = Emotion(emotions[i], mini_emotions[i], float(probs[i])) 
    return sentences

def parse_book(book_path, verbose = False):
    if verbose:
        print(f'===================Begin Parsing======================')
        start = time.time()
    with open(book_path, "r") as txtFile:
        text = txtFile.read()
        
    chunks = convert_text_to_chunks(text)
    
    with Pool(10) as p:
        pooled_opt = p.map(parse_into_sents_corefs,chunks)
        sentences = [ sentence for par,_ in pooled_opt for sentence in par]
        corefs = get_merged_corefs([ coref_dict for _,coref_dict in pooled_opt])

    if verbose:
        ckpt1 = time.time()
        print(f'Sentences and Coref obtained : {ckpt1-start}')
        
    batch_generator = generate_sentence_batches(sentences, BATCH_SIZE=8)

    emotion_batches = []
    for batch in batch_generator:
        emotion_batches.append(get_emotion_per_batch(batch))
    
    
    sentences = merge_emotions_to_sentences(sentences, emotion_batches)
    if verbose:
        ckpt2 = time.time()
        print(f'Emotions obtained : {ckpt2-ckpt1}')
    
    if verbose:
        print(f"\nSentences : {len(sentences)}, characters : {len(corefs.keys())}")
        end = time.time()
        print(f'Processing_time : {end-start}')
        print(f'===================End Parsing======================')
    return Book(book_path, text, sentences, corefs)
    