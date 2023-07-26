import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from fast_ctc_decode import beam_search, viterbi_search
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.device = device
        
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # Flatten the LSTM parameters before forward propagation
        # self.lstm.flatten_parameters()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out_all=self.fc(out)

        return out_all





class RNADataset(Dataset):
    def __init__(self, chunks, references, reference_lengths):
        self.chunks = chunks
        self.references = references
        self.reference_lengths = reference_lengths

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index], self.references[index], self.reference_lengths[index]


ctc_loss = nn.CTCLoss(blank=0)
alphabet = "NACGT"


def compute_ctc_loss(outputs, targets, target_lengths):
    # 去除填充值
    targets = targets[:, :target_lengths.max().item()].contiguous().long()
    targets=torch.argmax(targets, dim=-1)
    outputs_c = outputs.cpu().detach().numpy()
    seqs = [viterbi_search(slice, alphabet)[0] for slice in outputs_c]
    
    # 获取每个切片的长度，并将其转换为一个大小为 [32] 的长整型张量
    outputs_lengths = torch.LongTensor([len(seq) for seq in seqs])
    
    outputs = outputs.transpose(0, 1).to(torch.float32)
    
    # print(outputs.shape,targets.shape,outputs_lengths.shape,target_lengths.shape)
    loss = ctc_loss(outputs, targets, outputs_lengths, target_lengths)
    return loss



def train(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets, target_lengths) in enumerate(dataloader):
        inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32)
        # print(inputs.shape,targets.shape,inputs.device,targets.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i%50==0:
            print(i," loss:",loss)
    return total_loss / len(dataloader)


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    true_labels = []
    pred_probs = []
    with torch.no_grad():
        for i, (inputs, targets, target_lengths) in enumerate(dataloader):
            inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, target_lengths)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 2)
            predicted = predicted.transpose(0, 1)
            targets = targets[:, :target_lengths.max().item()].contiguous().long()
            mask = (targets != 0)
            print(predicted.shape,targets.shape)
            num_correct = torch.sum(torch.masked_select(predicted, mask) == torch.masked_select(targets, mask))
            total_correct += num_correct.item()
            true_labels += targets.masked_select(mask).tolist()
            pred_probs += torch.softmax(outputs.data, 2).max(2)[0].masked_select(mask).tolist()
    accuracy = total_correct / len(dataloader.dataset)
    auc = roc_auc_score(true_labels, pred_probs, multi_class='ovo')
    precision = precision_score(true_labels, [int(p >= 0.5) for p in pred_probs], average='macro')
    recall = recall_score(true_labels, [int(p >= 0.5) for p in pred_probs], average='macro')
    f1 = f1_score(true_labels, [int(p >= 0.5) for p in pred_probs], average='macro')
    return total_loss / len(dataloader), accuracy, auc, precision, recall, f1
