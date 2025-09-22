from Vocabulary import Vocabulary
import torch
import string

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

 def vectorize(self, textDoc):
  """
    Create a collapsed one-hot vector for a text document
    Args:
        textDoc (str): A text document
    Returns: one_hot (np.ndarray): the collapsed one-hot encoding
  """
  one_hot = torch.zeros(len(self.text_vocab), dtype=torch.float32)
  for token in textDoc.split(" "):
   if token not in string.punctuation:
    one_hot[self.text_vocab.lookup_token(token)] = 1;
  return one_hot

 @classmethod
 def from_serializable(cls, contents):
    """
        Initiate a TextVectorizer from a serializable dictionary
    Args:
        contents (dict): the serializable dictionary
    Returns:
        an instance of the TextVectorizer class
    """
    text_vocab = Vocabulary.from_serializable(contents['text_vocab'])
    return cls(text_vocab=text_vocab)
