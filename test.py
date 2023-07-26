from fast_ctc_decode import beam_search, viterbi_search
import numpy as np
alphabet = "NACGT"
posteriors = np.random.rand(100, len(alphabet)).astype(np.float32)

seq, path = viterbi_search(posteriors, alphabet)

# >>> seq
# 'ACACTCGCAGCGCGATACGACTGATCGAGATATACTCAGTGTACACAGT'

seq, path = beam_search(posteriors, alphabet, beam_size=5, beam_cut_threshold=0.1)

# >>> seq
# 'ACACTCGCAGCGCGATACGACTGATCGAGATATACTCAGTGTACACAGT'

print()






