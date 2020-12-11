from entities import *
import time, os, sys, re, json, collections
from multiprocessing import Pool
import string 

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

'''
Parse a mention and the sentence in which it occurs
in order to extract the patient, agent, and predicative
words used in relation to the entity mentioned.
'''
def parse_sent_and_mention(sent, mention, par_id):
    agents = []
    patients = []
    predicatives = []
    # Iterate over tokens in the mention
    for token in mention:
        token_tag = sent.token_tags[token.i - sent.global_token_start]
        # If the token's dependency tag is nsubj, find it's parent and set the lemma of this word to
        # be an agent of this entity.
        if token_tag.dep == 'nsubj':
            idx = token_tag.head_global_id - sent.global_token_start
            agent_verb = sent.token_tags[idx].lemma
            agents.append(Occurrence(agent_verb, sent.sentence_id, par_id, idx, idx+1))
            #print(" mention: ", mention, " token: ", token, " id ", token.i, "agent : ", agent_verb)
            
        # If the token's dependency tag is dobj or nsubjpass, find it's parent and set the lemma of this word to
        # be an patient of this entity.
        if (token_tag.dep ==  'dobj') or (token_tag.dep == 'nsubjpass'):
            idx = token_tag.head_global_id - sent.global_token_start
            patient_verb = sent.token_tags[idx].lemma
            patients.append(Occurrence(patient_verb, sent.sentence_id, par_id, idx, idx+1))
            #print(" mention: ", mention, " token: ", token, " id ", token.i, "patient : ", patient_verb)

    # Now we handle dependencies in the other direction to get predicatives.
    # 'man' is the predicative of 'Tim' in the sentence "Tim is a man."
    # Iterate over sentence tokens
    for token_tag in sent.token_tags:
        # Only consider tokens not in the mention:
        if not ((token_tag.token_global_id >= mention.start) and (token_tag.token_global_id <= mention.end)):
            # ignore punctuation
            if token_tag.pos != 'PUNCT':
                # Check if the parent of the word is a "be" verb (is, are, be, etc.)
                if sent.token_tags[token_tag.head_global_id - sent.global_token_start].lemma == "be":
                    to_be_verb = sent.token_tags[token_tag.head_global_id - sent.global_token_start]
                    # Check if the parent of the "be" verb is part of the mention
                    if (to_be_verb.head_global_id >= mention.start) and (to_be_verb.head_global_id <= mention.end):
                        idx = token_tag.token_global_id - sent.global_token_start
                        pred_word = sent.token_tags[idx].lemma
                        predicatives.append(Occurrence(pred_word, sent.sentence_id, par_id, idx, idx+1))
                        #print(" mention: ", mention, " token: ", token, " id ", token_tag.token_global_id,  "predicative : ", pred_word)
                    
    return agents, patients, predicatives
        

def parse_into_sentences_characters(inp):
    exclude = set(string.punctuation)
    text, par_id = inp
    doc = nlp(text)
    # parse into sentences
    sentences = []
    sentence_id_for_tokens = []
    for s, sent in enumerate(doc.sents):
        tokens = doc[sent.start:sent.end]
        sentence_id_for_tokens += [s] * len(tokens)
        token_tags = [TOKEN_TAGS(i, token.i, token.text, token.lemma_, token.pos_, token.pos_, token.dep_, token.head.i) for i, token in enumerate(tokens)]
        emotion_tags = Emotion(None,None,None)
        sentences.append(Sentence(s, par_id, sent.start, sent.text, token_tags, emotion_tags))
    
    if doc._.has_coref:
        corefs = {}
        for cluster in doc._.coref_clusters:
            # If an entry for this coref doesn't yet exist, create one
            main_name = cluster.main.text.lower()
            main_name = ''.join(ch for ch in main_name if ch not in exclude).strip()
    
            if not ( main_name in corefs):
                corefs[main_name] = {"mentions" : [], "agents" : [], "patients" : [], "preds" : []}
            # Update the entry with new mention and any parsed verbs or predicatives
            for mention in cluster.mentions:
                mention_name = mention.text.lower()
                mention_name = ''.join(ch for ch in main_name if ch not in exclude).strip()    
                mention_sent = sentence_id_for_tokens[mention.start]
                corefs[main_name]["mentions"].append(Occurrence(mention_name, mention_sent, par_id, mention.start, mention.end))
                agents, patients, preds = parse_sent_and_mention(sentences[mention_sent], mention, par_id)
                corefs[main_name]["agents"] += agents
                corefs[main_name]["patients"] += patients
                corefs[main_name]["preds"] += preds    

    return sentences, corefs



def get_merged_characters(coref_dicts, max_fuzz = 70):
    characters = []
    main_coref = {}
    for dict_ in coref_dicts:
        for k,v in dict_.items():
            if k in main_coref:
                main_coref[k].update(v)
            else:
                main_coref[k] = v
    
    merged_coref = {}
    char_counts = {}
    for k,v in main_coref.items():
        added = 0
        for merged_char in merged_coref.keys():
            if fuzz.partial_ratio(merged_char, k) > max_fuzz:
                merged_coref[merged_char].update(v)
                added = 1
                char_counts[merged_char]+=len(v)
                break
        if added==0:
            merged_coref[k] = v
            char_counts[k]=len(v)
            
    
    ranked_chars = sorted(char_counts, key=char_counts.get, reverse=True)
    for char in merged_coref:
        rank = ranked_chars.index(char) + 1
        character = Character(rank, char, merged_coref[char]['mentions'], merged_coref[char]['agents'], merged_coref[char]['patients'], merged_coref[char]['preds'])
        characters.append(character)
    
    return characters

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


<<<<<<< HEAD
def parse_book(book_path, verbose = False, poolNum=5):
    try:
        if verbose:
            print(f'===================Begin Parsing======================')
            print(book_path+"\n")
            start = time.time()
        with open(book_path, "r") as txtFile:
            text = txtFile.read()

        chunks = convert_text_to_chunks(text)

        p = Pool(poolNum)
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
        return p, Book(book_path, text, sentences, corefs)
    except Exception as e:
        if verbose:
            print(e)
            end = time.time()
            print("Error Parsing, Skipping")
            print(f'Processing_time : {end-start}')
            print(f'===================End Parsing======================')
        return p, None
=======
def parse_book(book_path, verbose = False, threads=5, batch_size=8):
    if verbose:
        print(f'===================Begin Parsing======================')
        start = time.time()
    with open(book_path, "r") as txtFile:
        text = txtFile.read()
        
    chunks = convert_text_to_chunks(text)
    
    with Pool(threads) as p:
        pooled_opt = p.map(parse_into_sentences_characters,chunks)
        sentences = [ sentence for par,_ in pooled_opt for sentence in par]
        characters = get_merged_characters([ coref_dict for _,coref_dict in pooled_opt])
        #get_verbs(corefs)

    if verbose:
        ckpt1 = time.time()
        print(f'Sentences and Coref obtained : {ckpt1-start}')
        
    batch_generator = generate_sentence_batches(sentences, BATCH_SIZE=batch_size)

    emotion_batches = []
    for batch in batch_generator:
        emotion_batches.append(get_emotion_per_batch(batch))
    
    
    sentences = merge_emotions_to_sentences(sentences, emotion_batches)
    if verbose:
        ckpt2 = time.time()
        print(f'Emotions obtained : {ckpt2-ckpt1}')
    
    if verbose:
        print(f"\nSentences : {len(sentences)}, characters : {len(characters)}")
        end = time.time()
        print(f'Processing_time : {end-start}')
        print(f'===================End Parsing======================')
    return Book(book_path, text, sentences, characters)
>>>>>>> refactor corefs into characters
    
