import os
import h5py
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from itertools import islice as take
import copy

path = "./Reads/"
output_directory = "./output"


def chunk_dataset(path, chunk_len, num_chunks=None):
    def all_chunks():
        for file_ in os.listdir(path):

            for file in os.listdir(os.path.join(path, file_)):

                if file.endswith('.fast5'):
                    path_ = os.path.join(path, file_)
                    with h5py.File(os.path.join(path_, file), 'r') as fast5_file:
                        data_path = fast5_file['Raw/Reads']
                        try:
                            for data in data_path:
                                if data.startswith("Read"):
                                    # Access the Signal dataset and convert to npy array
                                    signal = data_path[data]["Signal"][()]
                            for chunk, target in get_chunks(fast5_file, regular_break_points(len(signal), chunk_len)):
                                yield (chunk, target)
                        except KeyError:
                            continue

    all_chunks_gen = all_chunks()
    chunks, targets = zip(*tqdm(take(all_chunks_gen, num_chunks), total=num_chunks))
    targets, target_lens = pad_lengths(targets)  # convert refs from ragged arrray
    return ChunkDataSet(chunks, targets, target_lens)


def get_chunks(fast5_file, break_points):
    global array_start, array_target
    sample = scale(fast5_file)
    tmps = fast5_file["Analyses"].keys()
    Ref_to_signal = []
    Reference = []
    for tmp in tmps:
        if tmp.startswith('RawGenomeCorrected'):
            events = fast5_file["Analyses/" + tmp + "/BaseCalled_template/"]["Events"][()]
            for _, i in enumerate(events):
                Ref_to_signal.append(i[2])
                Reference.append(ACGT_2_num(i[4].decode()) + 1)

            array_start = np.stack(Ref_to_signal, axis=0)
            array_target = np.stack(Reference, axis=0)

    pointers = array_start
    target = array_target  # CTC convention
    return (
        (sample[i:j], target[ti:tj]) for (i, j), (ti, tj)
        in zip(break_points, np.searchsorted(pointers, break_points))
    )


def scale(fast5_file, normalise=True):
    """ scale and normalise a read """

    global scaled
    reads_group = fast5_file["Raw/Reads"]

    # Find the sample
    for group in reads_group:
        if group.startswith("Read"):
            # Access the Signal dataset and convert to npy array
            samples = reads_group[group]["Signal"][()]
            # scaled = (scaling * (samples + offset)).astype(np.float32)

    if normalise:
        tmps = fast5_file["Analyses"].keys()

        for tmp in tmps:

            if tmp.startswith('RawGenomeCorrected'):
                scale = fast5_file["Analyses/" + tmp + "/BaseCalled_template"].attrs["scale"]
                shift = fast5_file["Analyses/" + tmp + "/BaseCalled_template"].attrs["shift"]

                return (samples - shift) / scale

    return scaled


def ACGT_2_num(char):
    if char == 'A':
        return 0
    elif char == 'C':
        return 1
    elif char == 'G':
        return 2
    elif char == 'T':
        return 3
    else:
        raise ValueError('Invalid input')


# **********************************************************************************************************************************************************************
class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


def regular_break_points(n, chunk_len, overlap=0, align='mid'):
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {'left': 0, 'mid': remainder // 2, 'right': remainder}[align]
    starts = np.arange(start, start + num_chunks * (chunk_len - overlap), (chunk_len - overlap))
    return np.vstack([starts, starts + chunk_len]).T


def pad_lengths(ragged_array, max_len=None):
    lengths: ndarray = np.array([len(x) for x in ragged_array], dtype=np.uint16)
    padded = np.zeros((len(ragged_array), max_len or np.max(lengths)), dtype=ragged_array[0].dtype)
    for x, y in zip(ragged_array, padded):
        y[:len(x)] = x
    return padded, lengths


def typical_indices(x, n=2.5):
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n * sd < x) & (x < mu + n * sd))
    return idx


def filter_chunks(ds, idx):
    filtered = ChunkDataSet(ds.chunks.squeeze(1)[idx], ds.targets[idx], ds.lengths[idx])
    filtered.targets = filtered.targets[:, :filtered.lengths.max()]
    return filtered


def save_chunks(chunks, output_directory):
    a = chunks.chunks.squeeze(1)
    b = chunks.targets
    c = chunks.lengths
    indices = c != 0
    aa = np.compress(indices, a, axis=0)
    bb = np.compress(indices, b, axis=0)
    cc = np.compress(indices, c, axis=0)
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "chunks.npy"), aa)
    np.save(os.path.join(output_directory, "references.npy"), bb)
    np.save(os.path.join(output_directory, "reference_lengths.npy"), cc)
    print()
    print("> data written to %s:" % output_directory)
    print("  - chunks.npy with shape", aa.shape)
    print("  - references.npy with shape", bb.shape)
    print("  - reference_lengths.npy shape", cc.shape)




# training_chunks = chunk_dataset(path, 3600)
# training_indices = typical_indices(training_chunks.lengths)
# training_chunks = filter_chunks(training_chunks, np.random.permutation(training_indices))
# save_chunks(training_chunks, output_directory)

a = np.load('./output/chunks.npy')
b = np.load('./output/references.npy')
c = np.load('./output/reference_lengths.npy')
aa = np.load('C:/Users/sfy/Desktop/sequence_test/outputs/chunks.npy')
# bb = np.load('C:/Users/sfy/Desktop/sequence_test/outputs/references.npy')
# cc = np.load('C:/Users/sfy/Desktop/sequence_test/outputs/reference_lengths.npy')
print("ok")
