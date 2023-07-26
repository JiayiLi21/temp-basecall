from utils import LSTMModel, RNADataset, train, evaluate, compute_ctc_loss
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 仅使用第0、1个GPU
num_gpus=1
#torch.backends.cudnn.enabled=False
batch_size = 1
input_size = 3600  # 输入信号段长度
learning_rate = 10e-8
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

train_chunks = np.load('./preprocessed/train_chunks.npy')
train_references = np.load('./preprocessed/train_references.npy')
train_reference_lengths = np.load('./preprocessed/train_reference_lengths.npy')

valid_chunks = np.load('./preprocessed/valid_chunks.npy')
valid_references = np.load('./preprocessed/valid_references.npy')
valid_reference_lengths = np.load('./preprocessed/valid_reference_lengths.npy')

train_chunks = train_chunks.reshape(train_chunks.shape[0], train_chunks.shape[1], 1)
valid_chunks = valid_chunks.reshape(valid_chunks.shape[0], valid_chunks.shape[1], 1)

train_reference_lengths = train_reference_lengths.astype(np.int32)
valid_reference_lengths = valid_reference_lengths.astype(np.int32)

train_dataset = RNADataset(torch.tensor(train_chunks), torch.tensor(train_references),
                           torch.tensor(train_reference_lengths))
valid_dataset = RNADataset(torch.tensor(valid_chunks), torch.tensor(valid_references),
                           torch.tensor(valid_reference_lengths))

# test_dataset = RNADataset(torch.tensor(test_chunks), torch.tensor(test_references), torch.tensor(test_reference_lengths))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,drop_last=True)

# Model
input_size = 1  # 输入大小（A，C，G，T四个碱基）
hidden_size = 64  # 隐状态大小
num_layers = 2  # LSTM层数
num_classes = 5  # 类别数（四个碱基和一个空白标签）
model = LSTMModel(input_size, hidden_size, num_layers, num_classes,device).to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)


model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'lstm_model.pth')


if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f'Loaded model from {model_path}')

# Train

for epoch in range(num_epochs):
    
    # train_loss = train(model, optimizer, compute_ctc_loss, train_loader, device)

    valid_loss, valid_acc,valid_auc,valid_precision,valid_recall,valid_f1 = evaluate(model, compute_ctc_loss, valid_loader, device)
    scheduler.step(valid_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid AUC: {valid_auc:.3f}, Valid Precision: {valid_precision:.3f}, Valid Recall: {valid_recall:.3f}, Valid F1: {valid_f1:.3f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')