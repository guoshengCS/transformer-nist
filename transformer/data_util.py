import os
import tarfile
import glob

import random

class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"

class EndEpoch():
    pass

class Pool(object):
    def __init__(self, sample_generator, pool_size, sort):
        self._pool_size = pool_size
        self._pool = []
        self._sample_generator = sample_generator()
        self._end = False
        self._sort = sort

    def _fill(self):
        while len(self._pool) < self._pool_size and not self._end:
            try:
                sample = self._sample_generator.next()
                self._pool.append(sample)
            except StopIteration as e:
                self._end = True
                break

        if self._sort:
            self._pool.sort(
                    key=lambda sample: max(len(sample[0]), len(sample[1]))
                    if len(sample) > 1 else len(sample[0])
            )

        if self._end and len(self._pool) < self._pool_size:
            self._pool.append(EndEpoch())

    def push_back(self, samples):
        if len(self._pool) != 0:
            raise Exception("Pool should be empty.")

        if len(samples) >= self._pool_size:
            raise Exception("Capacity of pool should be greater than a batch. "
                            "Please enlarge `pool_size`.")

        for sample in samples:
            self._pool.append(sample)

        self._fill()

    def next(self, look=False):
        if len(self._pool) == 0:
            return None
        else:
            return self._pool[0] if look else self._pool.pop(0)

class DataReader(object):
    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 batch_size,
                 pool_size,
                 sort_type=SortType.NONE,
                 clip_last_batch=True,
                 tar_fname=None,
                 min_length=0,
                 max_length=100,
                 shuffle=True,
                 shuffle_batch=False,
                 use_token_batch=False,
                 delimiter='\t',
                 start_mark='<s>',
                 end_mark='<e>',
                 unk_mark='<unk>',
                 seed=0):
        self._src_vocab = self._load_dict(src_vocab_fpath)
        self._only_src = True
        if trg_vocab_fpath is not None:
            self._trg_vocab = self._load_dict(trg_vocab_fpath)
            self._only_src = False
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._delimiter = delimiter
        self._epoch_batches = []

        src_seq_words, trg_seq_words = self._load_data(fpattern, tar_fname)
        self._src_seq_ids = [[
            self._src_vocab.get(word, self._src_vocab.get(unk_mark))
            for word in ([start_mark] + src_seq_words + [end_mark])]
            for src_seq_words in src_seq_words
        ]

        self._sample_count = len(self._src_seq_ids)

        if not self._only_src:
            self._trg_seq_ids = [[
                self._trg_vocab.get(word, self._trg_vocab.get(unk_mark))
                for word in ([start_mark] + trg_seq + [end_mark])]
                for trg_seq in trg_seq_words
            ]
            if len(self._trg_seq_ids) != self._sample_count:
                raise Exception("Inconsistent sample count between "
                                "source sequences and target sequences.")
        else: self._trg_seq_ids = None

        self._sample_idxs = [i for i in xrange(self._sample_count)]
        self._sorted = False

        random.seed(seed)

    def _parse_file(self, f_obj):
        src_seq_words = []
        trg_seq_words = []

        for line in f_obj:
            fields = line.strip().split(self._delimiter)

            if len(fields) != 2 or (self._only_src and len(fields) != 1):
                continue

            sample_words = []
            is_valid_sample = True
            max_len = -1

            for i, seq in enumerate(fields):
                seq_words = seq.split()
                max_len = max(max_len, len(seq_words))
                if len(seq_words) == 0 or \
                        len(seq_words) < self._min_length or \
                        len(seq_words) > self._max_length or \
                        (self._use_token_batch and max_len > self._batch_size):
                    is_valid_sample = False
                    break

                sample_words.append(seq_words)

            if not is_valid_sample: continue

            src_seq_words.append(sample_words[0])

            if not self._only_src:
                trg_seq_words.append(sample_words[1])

        return (src_seq_words, trg_seq_words)

    def _load_data(self, fpattern, tar_fname):
        fpaths = glob.glob(fpattern)

        src_seq_words = []
        trg_seq_words = []

        if len(fpaths) == 1 and tarfile.is_tarfile(fpaths[0]):
            if tar_fname is None:
                raise Exception("If tar file provided, please set tar_fname.")

            f = tarfile.open(fpaths[0], 'r')
            part_file_data = self._parse_file(f.extractfile(tar_fname))
            src_seq_words = part_file_data[0]
            trg_seq_words = part_file_data[1]
        else:
            for fpath in fpaths:
                if not os.path.isfile(fpath):
                    raise IOError("Invalid file: %s" % fpath)

                part_file_data = self._parse_file(open(fpath, 'r'))
                src_seq_words.extend(part_file_data[0])
                trg_seq_words.extend(part_file_data[1])

        return src_seq_words, trg_seq_words

    def _load_dict(self, dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "r") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip()
                else:
                    word_dict[line.strip()] = idx
        return word_dict

    def _sample_generator(self):
        if self._sort_type == SortType.GLOBAL:
            if not self._sorted:
                self._sample_idxs.sort(
                    key=lambda idx: max(len(self._src_seq_ids[idx]),
                    len(self._trg_seq_ids[idx] if not self._only_src else 0))
                )
                self._sorted = True
        elif self._shuffle:
            random.shuffle(self._sample_idxs)

        for sample_idx in self._sample_idxs:
            if self._only_src:
                yield (self._src_seq_ids[sample_idx])
            else:
                yield (self._src_seq_ids[sample_idx],
                       self._trg_seq_ids[sample_idx][:-1],
                       self._trg_seq_ids[sample_idx][1:])

    def batch_generator(self):
        pool = Pool(self._sample_generator,
                    self._pool_size,
                    True if self._sort_type == SortType.POOL else False)

        def next_batch():
            batch_data = []
            max_len = -1
            batch_max_seq_len = -1

            while True:
                sample = pool.next(look=True)

                if sample is None:
                    pool.push_back(batch_data)
                    batch_data = []
                    continue

                if isinstance(sample, EndEpoch):
                    return batch_data, batch_max_seq_len, True

                max_len = max(max_len, len(sample[0]))

                if not self._only_src:
                    max_len = max(max_len, len(sample[1]))

                if self._use_token_batch:
                    if max_len * (len(batch_data) + 1) < self._batch_size:
                        batch_max_seq_len = max_len
                        batch_data.append(pool.next())
                    else:
                        return batch_data, batch_max_seq_len, False
                else:
                    if len(batch_data) < self._batch_size:
                        batch_max_seq_len = max_len
                        batch_data.append(pool.next())
                    else:
                        return batch_data, batch_max_seq_len, False

        if not self._shuffle_batch:
            batch_data, batch_max_seq_len, last_batch = next_batch()
            while not last_batch:
                yield batch_data
                batch_data, batch_max_seq_len, last_batch = next_batch()

            batch_size = len(batch_data)
            if self._use_token_batch:
                batch_size *= batch_max_seq_len

            if (not self._clip_last_batch and len(batch_data) > 0) \
                    or (batch_size == self._batch_size):
                yield batch_data
        else:
            # should re-generate batches
            if self._sort_type == SortType.POOL \
                    or len(self._epoch_batches) == 0:
                self._epoch_batches = []
                batch_data, batch_max_seq_len, last_batch = next_batch()
                while not last_batch:
                    self._epoch_batches.append(batch_data)
                    batch_data, batch_max_seq_len, last_batch = next_batch()

                batch_size = len(batch_data)
                if self._use_token_batch:
                    batch_size *= batch_max_seq_len

                if (not self._clip_last_batch and len(batch_data) > 0) \
                        or (batch_size == self._batch_size):
                    self._epoch_batches.append(batch_data)

            random.shuffle(self._epoch_batches)

            for batch_data in self._epoch_batches:
                yield batch_data

if __name__ == "__main__":
    '''data_loader = DataLoader("/root/workspace/unify_reader/wmt16/en_10000.dict",
                             "/root/workspace/unify_reader/wmt16/de_10000.dict",
                             "/root/workspace/unify_reader/wmt16/wmt16.tar.gz",
                             2, tar_fname="wmt16/train")'''

    data_loader = DataReader(
        "/root/workspace/unify_reader/nist06n_tiny/cn_30001.dict.unify",
        "/root/workspace/unify_reader/nist06n_tiny/en_30001.dict.unify",
        "/root/workspace/unify_reader/nist06n_tiny/data/part-*",
        30,
        100000,
        sort_type=SortType.GLOBAL,
        use_token_batch=True)
    for batch_data in data_loader.batch_generator():
        print batch_data
        break
