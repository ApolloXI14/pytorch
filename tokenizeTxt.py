#TODO: Expand logic to process CSVs using pandas.DataFrame.loc
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# Preliminary program to tokenize a single text file for eventual ML text encoding

import argparse
import os
import re
from io import open
import tokenize
import pandas as pd

import spacy

filePath = ''


def preprocess_text(text):
  text = re.sub(r"([.,!?])", r" \1 ", text)
 # remove all non-alphanumeric characters
  text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
  return text

def main(filepath=""):
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
  dict = {}
  if ".csv" in filepath:
   print(f'pandas functionality needs to be implemented here to read {filepath}');
   csv_file = pd.read_csv(filepath);
   print(csv_file);
   csv_train = csv_file.train;
  for file in range(0,len(csv_train)):
   print(csv_train[file]);
   #open(csv_train[file].strip()).read();
   print(open(csv_train[file].strip()).read());
  if ".txt" in filepath:
   print('.txt detected')
   text = open(filepath, "r", encoding="utf-8").read().lower();
   print(text)
   # TODO: Text can also be processed as bytes, which can support all
   # languages seamlessly; explore later
   #text = nlp(open(filePath, "rb").read().lower())
   text = preprocess_text(text); 
   text = nlp(text); 
   tokens = [str(token) for token in text]
   for li, token in enumerate(tokens):
    # If token is NOT in dictionary, only then add, to prevent
    # skipping IDs on keys that appear more than once
    currentDictLen = len(dict.items());
    if token not in dict:
     dict[token] = currentDictLen + 1;
  return dict

if __name__ ==  "__main__":
 parser = argparse.ArgumentParser();
 parser.add_argument('-f', '--filepath', help="Filepath of file to be read and tokenized");
 args = parser.parse_args();
 filepath = args.filepath;
 #test = open(filePath, "r", encoding='utf-8').read().split('\n')
 main(filepath)
	#main(FILEPATH)

