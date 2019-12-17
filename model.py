import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import sys
import re


REPLACE_NO_SPACE = re.compile("[*.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

STOPWORDS = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while",
             "above", "both", "up", "to", "ours", "had", "she", "all", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
APPOS = {
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'd": "i had",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "i have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": "are",
    "wasn't": "was not",
    "we'll": "we will",
    "didn't": "did not"
}


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = tnn.LSTM(50, 150, batch_first=True,
                             num_layers=2, bidirectional=True, dropout=0.35)
        self.dropout = tnn.Dropout(p=0.15)
        self.l1 = tnn.Linear(150, 64)
        self.l2 = tnn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        x = tnn.utils.rnn.pack_padded_sequence(
            input, length, batch_first=True, enforce_sorted=False)
        _, (x, _) = self.lstm(x)
        x = self.dropout(x)
        x = self.l1(x[-2])
        x = tnn.functional.relu(x)
        x = self.l2(x)
        x = x.squeeze(0).squeeze(-1)
        return x


class PreProcessing():

    def pre(x):
        """Called after tokenization"""
        def stemmatize(x):
            if len(x) < 3:
                return x
            # noun
            if len(x) > 6 and x[-4:] == 'ness':
                x = x[:-4]
            elif len(x) > 5 and x[-3:] == 'ful':
                x = x[:-3]
            elif len(x) > 7 and x[-5:] == 'fully':
                x = x[:-5]
            # plural to singular
            if len(x) > 4 and x[-2:] == 'es' and (x[-3] in ['s', 'x', 'z'] or (x[-4] + x[-3]) in ['sh', 'ch', 'ss']):
                x = x[:-2]
            elif x[-1] == 's' and x[-2] not in ['a', 'e', 'i', 'o', 'u']:
                x = x[:-1]
            if len(x) > 3 and x[-1] == 'e':
                x = x[:-1]
            elif len(x) > 3 and (x[-2:] == 'ed' or x[-2:] == 'ly'):
                x = x[:-2]
            elif len(x) > 5 and x[-3:] == 'ing' and x[-4] == x[-5]:
                x = x[:-4]
            elif len(x) > 4 and (x[-3:] == 'ion' or x[-3:] == 'ing'):
                x = x[:-3]
            return x
        x = [stemmatize(tok) for tok in x]
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""

        return batch

    def tokenize(text):
        text = text.split()
        text = [APPOS[word] if word in APPOS else word for word in text]
        text = " ".join(text)
        text = REPLACE_NO_SPACE.sub("", text)
        text = REPLACE_WITH_SPACE.sub(" ", text)
        return text.split()

    text_field = data.Field(lower=True, include_lengths=True,
                            batch_first=True, preprocessing=pre, postprocessing=post, tokenize=tokenize, stop_words=STOPWORDS)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField,
                             train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)

    criterion = lossFunc()
    # Minimise the loss using the Adam algorithm.
    optimiser = topti.Adam(net.parameters(), lr=0.001)

    for epoch in range(20):
        running_loss = 0
        is_vanish = False
        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / 32))
                if (running_loss / 32) < 0.25:
                    is_vanish = True
                running_loss = 0
        if is_vanish:
            break

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()
