from torch.utils.data import Dataset
from Vectorize import TextVectorizer
import pandas as pd

class TextDataset(Dataset):
 def __init__(self, text_df, vectorizer):
  """
  Args:
    text_df (pandas.DataFrame): the dataset
    vectorizer (TextVectorizer): vectorizer instantiated
    from dataset
  """
  self.text_df = text_df;
  self._vectorizer = vectorizer;

  self.train_df = self.text_df[self.text_df.split=="train"];
  self.train_size = len(self.train_df);
  
  self.val_df = self.text_df[self.text_df.split=="val"];
  self.val_size = len(self.val_df);

  self.test_df = self.text_df[self.text_df.split=="test"];
  self.test_size = len(self.test_df);

  self._lookup_dict = {'train': {self.train_df, self.train_size},
  'val': {self.val_df, self.val_size}, 'test': {self.test_df,
  self.test_size} }
  self.set_split('train')

 @classmethod
 def load_dataset_and_make_vectorizer(cls, text_csv):
  """
  Load dataset and make a new vectorizer from scratch
  Args:
   text_csv (str): location of dataset
  Returns:
    an instance of TextDataset
  """
  text_df = pd.read_csv(text_csv);
  return cls(text_df, TextVectorizer.from_dataframe(text_df))

 def get_vectorizer(self):
  """
  Returns the vectorizer
  """
  return self._vectorizer;

 def set_split(self, split="train"):
  """
  Selects the splits in the dataset using a column in dataframe
  Args:
    split (str): one of "train", "val", or "test"
  """
  self._target_split = split;
  self._text_df, self._target_size = self._lookup_dict[split]

 def __len__(self):
  return self._target_size;

 def __getitem__(self, index):
  """
  the primary entry point method for PyTorch datasets
  Args:
    index (int): the index to the data point
  Returns:
    a dict of the data point's features (x_data) and label (y_target)
  """
  row = self._target_df.iloc[index];
  text_vector = self._vectorizer.vectorize(row.text);
  vocab_index = self._vectorizer.text_vocab.lookup_token(row.text);

  return {'x_data': text_vector, 'y_data': vocab_index }

 def get_num_batches(self, batch_size):
  """
  Given a batch size, return the number of batches in the dataset
  Args:
    batch_size (int)
  Returns:
    number of batches in the dataset
  """
  return len(self) # batch size
