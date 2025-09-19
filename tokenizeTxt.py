# Preliminary program to tokenize a single text file for eventual ML text encoding

import os
from io import open
import tokenize

import spacy

FILEPATH = "FILE_TO_TOKENIZE"

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
  text = open(filePath, "r", encoding="utf-8").read()
  tokens = [str(token) for token in nlp(text.lower())]
  print(tokens)

if __name__ ==  "__main__":
	main(FILEPATH)

