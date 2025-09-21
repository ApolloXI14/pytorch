
class Vocabulary(object):
 """Class to process text and extract vocab for mapping"""
 def __init__(self, token_to_idx=None, add_unk=True,
 unk_token="<UNK>"):
  """
    Args:
     token_to_idx (dict): pre-existing map of tokens to indices
     add_unk (bool): flag that indicates whether to add UNK token
     unk_token (str): the UNK token to add into Vocab
  """

  # If Vocab is blank, empty object
  if token_to_idx is None:
   token_to_idx = {};
  self._token_to_idx = token_to_idx;
  # Create dict of "idx: token" mappings for all tokens in
  # token_to_idx
  self._idx_to_token = { idx: token for token, idx in
  self._token_to_idx.items() }

  # unk logic if word is unknown
  self._add_unk = add_unk
  self._unk_token = unk_token
  self.unk_index = -1
  if add_unk:
   self.unk_index = self.add_token(unk_token)

  def to_serializable(self):
   """ returns a dictionary that can be serialized"""
   return { 'token_to_index': self._token_to_idx,
    'add_unk': self._add_unk,
    'unk_token': self._unk_token}

  @classmethod
  def from_serializable(cls, contents):
   """ inits Vocab from a serialized dictionary """
   return cls(**contents)

 def add_token(self, token):
   """ Update mapping dictionaries based on token 
   Args:
   token (str): the item to add into the Vocabulary
   Returns
   index (int): the integer corresponding to the token
   """
  # If token is already in vocab, just get index
   if token in self._token_to_idx:
    index = self._token_to_idx[token]
   else: # else create new dict entries
    index = len(self._token_to_idx)
    self._token_to_idx[token] = index
    self._idx_to_token[index] = token
   return index;
   
def lookup_token(self, token):
 """ Retrieve the index associated with the token
    of the UNK index if token isn't present
  Args:
    token (str): the token to look up
  Returns:
    index (int): the index corresponding to the token
  Note:
    'unk_index' needs to be >=0 (having been into Vocab)
    for the UNK functionality
 """
 if self.add_unk:
  return self._token_to_idx.get(token, self.unk_index)
 else:
  return self._token_to_idx[token]

 def lookup_index(self, index):
   """ Return the token associated with the index
   Args:
   index (int): the index to look up
   Returns:
   token (str): the token corresponding to index
   Raises:
   KeyError: if index is not in Vocab
   """
   if index not in self._idx_token:
    raise KeyError("the index (%d) is not in the Vocabulary" % index)
   return self._idx_to_token[index]
  
 def __str__(self):
  return "<Vocabulary(size=%d)>" % len(self)
 
 def __len__(self):
  return len(self._token_to_idx)
