import torch.nn as nn
import torch.nn.functional as F

class TextTransformer(nn.Module):
 """
 a multilayer perceptron (MLP) which transforms and generates text
 """
 def __init__(self, input_dim, hidden_dim, output_dim):
  """
  Args:
    num_features (int): the size of the input feature vector
  """
  super(TextTransformer, self).__init__();
  # "fc" = fully connected (nn layer)
  self.fc1 = nn.Linear(input_dim, hidden_dim)
  self.fc2 = nn.Linear(hidden_dim, output_dim)

 def forward(self, x_in, apply_softmax=False):
  """
  Args:
   x_in (torch.Tensor): an input data tensor
   x_in.shape should be (batch, input dim)
   apply_softmax (bool): a flag for the softmax activation
   Returns: resulting tensor; tensor.shape should be (batch,
   output_dim)
  """ 
  intermediate = F.relu(self.fc(x_in))
  output = self.fc2(intermediate);
   
  if apply_softmax:
   output = F.softmax(output, dim=1);
  return output;
