##### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from nlp_helpers import *
from model import SequenceClassifierNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


##### LOAD DATA
FILE = "datasets/intents.json"
data = load_classifier_data(FILE)
vocab = LxtClassifierVocab()
vocab.create_vocab(data)
X_train, Y_train = load_classifier_X_Y_train(data, vocab)

##### HYPERPARAMETERS
device = torch.device("cuda")
num_epochs = 500
ntoken = vocab.ntoken
num_labels = len(vocab.lbl_to_ix)
hidden_combos = [
    {"dim1": 256, "dim2": 128, "dim3": 64},
]
#hidden_combos = [{"dim1": 64, "dim2": 32, "dim3": 16}]
batch_sizes = [1]
learning_rates = [0.0001]

##### GRID SEARCH LOOPS
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for hidden_sizes in hidden_combos:
            model = SequenceClassifierNN(ntoken, num_labels, hidden_sizes).to(device)
            optimizers = list()
            optimizers.append(Adam(model.parameters(), lr=learning_rate))
            for optim in optimizers:
                ##### load model training variables 
                hidden_sizes_string = "hidden_sizes_" +str(hidden_sizes["dim1"])+"_"+str(hidden_sizes["dim2"])+"_"+str(hidden_sizes["dim3"])
                save_name = f"classifier_v1"
                dataset = NlpDataset(X_train, Y_train)
                train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
                criterion = nn.CrossEntropyLoss()
                tb = SummaryWriter(f"runs/classifier/{save_name}")
                ##### TRAINING LOOP
                print()
                print(f"TRAINING with model: {save_name}")
                for epoch in range(0, num_epochs+1):
                    epoch_loss = 0.0
                    for (input_batch, target_batch) in train_loader:
                        # init forward pass
                        input_batch = input_batch.to(device)
                        target_batch = target_batch.to(device)
                        optim.zero_grad()
                        # forward pass
                        output = model(input_batch)
                        loss = criterion(output, target_batch)
                        # backward
                        loss.backward()
                        optim.step()
                        # for printing
                        epoch_loss+=loss.item()
                    # print loss
                    if epoch % 5 == 0:
                        tb.add_scalar("Training Loss", epoch_loss/len(train_loader), epoch)
                        print(f"::::: loss for epoch {epoch}/{num_epochs}: {epoch_loss/len(train_loader)}")
                ##### POST TRAINING
                tb.close()
                training_data = {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "optim_type": type(optim),
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_sizes": hidden_sizes
                }
                save_file = f"saved_data/classifier/{save_name}.pth"
                torch.save(training_data, save_file)

                        
                        








