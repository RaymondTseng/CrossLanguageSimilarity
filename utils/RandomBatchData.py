import numpy as np
class RandomBatchData:
    def __init__(self, sources, targets, scores, sources_length, targets_length, batch_size):
        self.index = 0
        self.sources = sources
        self.targets = targets
        self.scores = scores
        self.sources_length = sources_length
        self.targets_length = targets_length
        self.batch_size = batch_size
        self.random_idx = range(len(sources))

    def next_batch(self):
        finish = None
        start = self.index
        end = start + self.batch_size
        if end > len(self.sources):
            end = len(self.sources)
            self.index = 0
            finish = True
        else:
            self.index += self.batch_size
            finish = False
        source_batch = []
        target_batch = []
        score_batch = []
        source_length_batch = []
        target_length_batch = []
        for i in range(start, end):
            source_batch.append(self.sources[self.random_idx[i]])
            target_batch.append(self.targets[self.random_idx[i]])
            score_batch.append(self.scores[self.random_idx[i]])
            source_length_batch.append(self.sources_length[self.random_idx[i]])
            target_length_batch.append(self.targets_length[self.random_idx[i]])
        if finish:
            np.random.shuffle(self.random_idx)
        return source_batch, target_batch, score_batch, source_length_batch, target_length_batch, finish