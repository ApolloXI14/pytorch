# Preliminary program to tokenize a single text file for eventual ML text encoding

import os
import re
from io import open
import tokenize

import spacy

FILEPATH = "FILEPATH HERE"



def preprocess_text(text):
  text = re.sub(r"([.,!?])", r" \1 ", text)
 # remove all non-alphanumeric characters
  text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
  return text

def main(filePath=""):
 #test = open(filePath, "r", encoding='utf-8').read().split('\n')
 #test = open(filePath, "rb").read()
 #for line in test:
 # print(line)
 #for li, line in enumerate(test):
 # print(li)

 #with tokenize.open(filePath) as f:
 # tokens = tokenize.generate_tokens(f.readline)
 # for token in tokens:
 #  #print(token)
 #  print(token.string)


  tokens = []
  nlp = spacy.load("en_core_web_sm");
  text = open(filePath, "r", encoding="utf-8").read().lower();
  # TODO: Text can also be processed as bytes, which can support all
  # languages seamlessly; explore later
  #text = nlp(open(filePath, "rb").read().lower())
  text = preprocess_text(text); 
  text = nlp(text); # tokenize text
  tokens = [str(token) for token in text]
  #print(tokens)
  dict = {}
  for li, token in enumerate(tokens):
    dict[li] = token;
    print(dict)
  return dict

if __name__ ==  "__main__":
	main(FILEPATH)

