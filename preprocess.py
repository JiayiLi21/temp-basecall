from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils import RNADataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
chunks = np.load('./output/chunks.npy')
chunks = (chunks - np.min(chunks)) / (np.max(chunks) - np.min(chunks))

references = np.load('./output/references.npy')
reference_lengths = np.load('./output/reference_lengths.npy')

num_classes = 5  # 四个碱基类型和一个空白标签
references = F.one_hot(torch.LongTensor(references), num_classes=num_classes).numpy()


train_chunks, valid_chunks, train_references, valid_references, train_reference_lengths, valid_reference_lengths = train_test_split(chunks, references, reference_lengths, test_size=0.1)
# valid_chunks, test_chunks, valid_references, test_references, valid_reference_lengths, test_reference_lengths = train_test_split(valid_chunks, valid_references, valid_reference_lengths, test_size=0.5, random_state=20)
output_directory="./preprocessed/"
output_directory = "./preprocessed/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
np.save(os.path.join(output_directory, "train_chunks.npy"), train_chunks)
np.save(os.path.join(output_directory, "train_references.npy"), train_references)
np.save(os.path.join(output_directory, "train_reference_lengths.npy"), train_reference_lengths)

np.save(os.path.join(output_directory, "valid_chunks.npy"), valid_chunks)
np.save(os.path.join(output_directory, "valid_references.npy"), valid_references)
np.save(os.path.join(output_directory, "valid_reference_lengths.npy"), valid_reference_lengths)