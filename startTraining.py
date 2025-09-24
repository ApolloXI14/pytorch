# validate "trainingArgs" before running
from TextDataset import TextDataset
from Transformer import TextTransformer
import torch.optim as optim
from trainingArgs import args
from DataLoader import generate_batches

def make_train_state(args):
 return {'epoch_index': 0,
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'test_loss': -1,
    'test_acc': -1}

train_state = make_train_state(args)

args.cuda = True;
if not torch.cuda_is_available():
 args.cuda = False;
args.device = torch.device("cuda" if args.cuda else "cpu")

#dataset and vectorizer
dataset =
TextDataset.load_dataset_and_make_vectorizer(args.model_state_file)
vectorizer = dataset.get_vectorizer();

#model
transformer = TextTransformer(3, 300, 4)
transformer = transformer.to(args.device)

#loss and optimizer
loss_func = Transformer.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.Adam(transformer.parameters(),
lr=args.learning_rate)

def compute_acc(datasetName):
 """
 Compute the accuracy after loss function execution
 """
 acc_batch = compute_accuracy(y_pred,
 batch_dict['y_target'])
 running_acc += (acc_batch - running_acc) / (batch_index + 1)
 train_state[datasetName + '_loss'].append(running_loss)
 train_state[datasetName + '_acc'].append(running_acc)


for epoch_index in range(args.num_epochs):
 train_state['epoch_index'] = epoch_index

 # Iterate over training dataset
 #setup: batch generator, set loss and acc to 0, set train mode on
 for datasetName in ['train', 'val']:
  dataset.set_split(datasetName)

  batch_generator = generate_batches(dataset,
  batch_size=args.batch_size, device=args.device)
  running_loss = 0.0
  running_acc = 0.0
  transformer[datasetName]()

  for batch_index, batch_dict in enumerate(batch_generator):
   #the step routine is 5 steps
   if datasetName == 'train': # training dataset steps
    optimizer.zero_grad() #1. zero the gradients
    y_pred = transformer(x_in=batch_dict['x_data'].float()) # 2. compute the output
    # 3. compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'])
    loss_batch = loss.to(args.device).item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)
    loss.backward() # 4. use loss to produce gradients
    optimizer.step() # 5. use optimizer to take gradient step
    compute_acc(datasetName) # final: training & val: compute the accuracy
   else: # validate dataset steps
    y_pred = transformer(x_in=batch_dict['x_data'].float()) # 1. compute the output
    #step 2. compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_batch = loss.item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)
    compute_acc(datasetName) # final: training & val: compute the accuracy
   
  
