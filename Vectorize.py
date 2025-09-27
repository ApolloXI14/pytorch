from Vocabulary import Vocabulary
import torch
import string
from tokenizeTxt import main as tokenizeTxt

class TextVectorizer(object):
 """
    A Vectorizer which converts a dictionary of words into a
    vector
 """
 def __init__(self, textDict):
  """
    Args:
        textDict (dict): a dictionary of id:word key value pairs
  """
  self.text_vocab = textDict;

 def vectorize(self, ngram):
  """
    Create a collapsed one-hot vector for an ngram
    Args:
        ngram (str): A piece of text
    Returns: one_hot (np.ndarray): the collapsed one-hot encoding
  """
  one_hot = torch.zeros(len(self.text_vocab), dtype=torch.float32)
  for token in ngram.split(" "):
   if token not in string.punctuation:
    one_hot[self.text_vocab.lookup_token(token)] = 1;
  return one_hot

 @classmethod
 def from_dataframe(cls, text_df):
  """
    Instantiate the vectorizer from the dataset dataframe
    Args:
        text_df (pandas.DataFrame):  the text dataset
    Returns:
        an instance of TextVectorizer
    Note:
        Pandas Dataframes are interchangeable with dictionaries
        or 2D arrays
  """
  text_vocab = Vocabulary(add_unk=True) # init an empty vocab dict
  # go through csv list of text files to turn into dicts
  for txtDoc in text_df: 
   textDict = tokenizeTxt(txtDoc);
   # inner loop to append "textDict" into text_vocab
   for word in textDict:
    text_vocab.add_token(textDict[word])
  #print(f'textDict: {text_vocab}')
  return cls(text_vocab)

 @classmethod
 def from_serializable(cls, contents):
    """
        Initiate a TextVectorizer from a serializable dictionary
    Args:
        contents (dict): the serializable dictionary
    Returns:
        an instance of the TextVectorizer class
    """
    text_vocab = Vocabulary.from_serializable(contents)
    #text_vocab = Vocabulary.from_serializable(contents['text_vocab'])
    return cls(text_vocab=text_vocab)

 def to_serializable(self):
  """ Create the serializable dictionary for caching

  Returns:
    contents (dict): the serializable dictionary
  """
  return { 'text_vocab': self.text_vocab.to_serializable() }
