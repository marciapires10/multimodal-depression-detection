import numpy as np
import pandas as pd
import torch
import transformers
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification


df = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")
# print(df.head())

X = df.Transcript
y = df.PHQ8_Binary

# split dataset into train, validation and test sets
def train_test_val(X, y, test="../test_split_Depression_AVEC2017.csv", val="../dev_split_Depression_AVEC2017.csv"):
    test_participants = pd.read_csv(test)['participant_ID'].values
    val_participants = pd.read_csv(val)['Participant_ID'].values
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    for i in range(y.shape[0]):
        participant_no = y.index[i]

        if participant_no in test_participants:
            X_test.append(X[participant_no])
            y_test.append(y[participant_no])
        elif participant_no in val_participants:
            X_val.append(X[participant_no])
            y_val.append(y[participant_no])
        else:
            X_train.append(X[participant_no])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(X_val), np.array(y_train), np.array(y_test), np.array(y_val)


X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(X, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("Number of total samples: ", len(X))
print("Number of train samples: ", len(X_train))
print("Number of validation samples: ", len(X_val))
print("Number of test samples: ", len(X_test))


def count_depression(data):
    n_depressed = 0
    depressed = 0
    final = []
    for s in data:
        if s == 0:
            n_depressed += 1
        else:
            depressed += 1

    final.append(n_depressed)
    final.append(depressed)

    return final


count_train = count_depression(y_train)
print("Depression distribution on train dataset: (ND/D)", count_train)

# need to balance the train dataset
X_train = X_train.reshape(-1, 1)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train, y_train = undersample.fit_resample(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_train = X_train.squeeze(1)

count_train = count_depression(y_train)
print("Depression distribution on train dataset AFTER balancing: (ND/D)", count_train)


# Followed the following tutorial for BERT: https://medium.com/analytics-vidhya/a-gentle-introduction-to-implementing-bert-using-hugging-face-35eb480cff3


# load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# tokenize and encode the data
def encode(data, tokenizer):
    input_ids = []
    attention_mask = []
    for text in data:
        tokenized_text = tokenizer.encode_plus(text,
                                               max_length=128,
                                               add_special_tokens=True,
                                               pad_to_max_length=True,
                                               return_attention_mask=True)
        input_ids.append(tokenized_text['input_ids'])
        attention_mask.append(tokenized_text['attention_mask'])

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


# get batches: convert lists to tensors, wrap tensors, create samplers and return final dataloader
def get_batches(x, y, tokenizer, batch_size):

    y = torch.tensor(list(y), dtype=torch.long)
    input_ids, attention_mask = encode(x, tokenizer)
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y)
    tensor_randomsampler = torch.utils.data.RandomSampler(tensor_dataset)
    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, sampler=tensor_randomsampler, batch_size=batch_size)
    return tensor_dataloader


train_dataloader = get_batches(X_train, y_train, tokenizer, batch_size=2)
val_dataloader = get_batches(X_val, y_val, tokenizer, batch_size=2)
test_dataloader = get_batches(X_test, y_test, tokenizer, batch_size=2)


# define parameters for the model
epochs = 2
parameters = {
    'learning_rate': 1e-5,
    'num_warmup_steps': 1000,
    'num_training_steps': len(train_dataloader) * epochs,
    'max_grad_norm': 1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

# define the model being used
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True, output_attentions=True)
model.to(device)

# define the optimizer
optimizer = transformers.AdamW(model.parameters(), lr=parameters['learning_rate'], correct_bias=False)

# define the scheduler
scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=parameters['num_warmup_steps'],
                                                         num_training_steps=parameters['num_training_steps'])


# function to train the model
def train_model(train_dataloader, model, optimizer, scheduler, epochs, device):
    model.train()  # training mode
    train_loss = 0
    total = 0

    #for e in range(epochs):
    for step, batch in enumerate(train_dataloader):
        batch = (t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # if i % 100 == 0:
        #     print("loss - {0}, iteration - {1}/{2}".format(loss, e + 1, i))

        model.zero_grad()
        optimizer.zero_grad()

        # calculate loss
        train_loss = train_loss + loss.item()
        total = total + 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['max_grad_norm'])

        # update the gradients
        optimizer.step()

        # update the learning rate
        scheduler.step()

    avg_train_loss = train_loss/total

    return avg_train_loss


# function to evaluate the validation set
def evaluate_val(val_dataloader, model, device):
    # predictions, true_labels = [], []
    val_loss = 0
    total = 0

    model.eval()
    for step, batch in enumerate(val_dataloader):
        batch = (t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            # logits = logits.cpu()
            prediction = torch.argmax(logits, dim=1)
            # true_label = labels.cpu().tolist()
            # predictions += prediction
            # true_labels += true_label

            val_loss = val_loss + loss.item()
            total = total + 1

    avg_val_loss = val_loss / total

    return avg_val_loss


# function to evaluate the final model test
def evaluate_test(test_dataloader, model, device):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch = (t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            logits = logits.cpu()

            predictions = np.argmax(logits, axis=1).tolist()
            target = labels.cpu().tolist()
            y_true.extend(target)
            y_pred.extend(predictions)

    return print(classification_report(y_true, y_pred))


train_losses, valid_losses = [], []
for e in range(epochs):
    train_loss = train_model(train_dataloader, model, optimizer, scheduler, epochs, device)
    valid_loss = evaluate_val(val_dataloader, model, device)

    print(f'Train loss: {train_loss:.3f}\tValidation loss: {valid_loss:.3f}\n')

    # if len(valid_losses) > 2 and all(valid_loss > loss for loss in valid_losses[-3:]):
    #     print('Stopping early')
    #     break
    #
    # train_losses.append(train_loss)
    # valid_losses.append(valid_loss)


evaluate_test(test_dataloader, model, device)

