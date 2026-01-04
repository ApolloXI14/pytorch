# Natural Language Processing with PyTorch

The purpose of this repository is to build a natural
language processor and recurrent neural network (RNN) that can receive
text as input and generate text as output.

The ultimate goal is to vectorize text ("Vectorize.py") through the
following steps.

1. Tokenize a text document and return a word dictionary
   (tokenizeTxt.py)
2. Wrap the returned word dictionary in a "Vocabulary" class (Vocabulary.py), which represents the entire vocabulary of all tokens in a text
3. Pass the "Vocabulary" class instance into the "TextVectorizer"
   class ("Vectorize.py") that will accept text input through the
   "vectorize" class method and return a one-hot vector based on the
   vocabulary of the initial text document matched against the input.
   E.g. A TextVectorizer class instance of a text containing "Hello world" that has the
   word "Hello" passed to its "vectorize" class method will return a
   vector of [1., 0]

The tokenizer ("tokenizeTxt.py") is the most important file of the entire procedure. A dictionary of tokens provides the basis for Vectorizing a TextDataset for training.

The key files in this procedure are "Vectorize.py" (which has two
dependencies, "tokenizeTxt" to create a python dictionary of all
"tokens" in a document, and "Vocabulary" which takes a python
dictionary and wraps it into a "Vocabulary" class), "TextDataset.py"
(Vectorize dependency) which takes a CSV file 
