import random
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from collections import defaultdict
import sys

class RawTimeSeriesDataset(data.Dataset):
    def __init__(self, dataset_dir, mode, name, ratio_number, ind_number):
        super(RawTimeSeriesDataset, self).__init__()
        self.dataset_dir = Path(dataset_dir)
        save_path = self.dataset_dir / name / str(ratio_number) / str(ind_number) / f"{name}.npy"
        dictionary = np.load(save_path, allow_pickle=True).item()
        
        raw_X = dictionary[f'X_{mode}']
        if isinstance(raw_X, np.ndarray):
            self.data = torch.FloatTensor(raw_X)
            if len(self.data.shape) == 2:
                self.data = self.data.unsqueeze(1)  # Shape: [N, 1, L] for univariate
        else:  # Assume list of np arrays for multivariate, each (C, L_i)
            if len(raw_X) > 0 and isinstance(raw_X[0], np.ndarray):
                C = raw_X[0].shape[0]
                max_L = max([x.shape[1] for x in raw_X])
                padded = []
                for x in raw_X:
                    pad_len = max_L - x.shape[1]
                    pad = np.zeros((C, pad_len), dtype=np.float32)
                    padded_x = np.concatenate([x, pad], axis=1)
                    padded.append(padded_x)
                self.data = torch.FloatTensor(np.stack(padded, axis=0))  # Shape: [N, C, max_L]
            else:
                raise ValueError("Invalid data format for multivariate")
        
        self.label = torch.LongTensor(dictionary[f'y_{mode}'])

        self.n_classes = len(set(self.label.tolist()))
        self.sequence_length = self.data.shape[-1]

        print(f"Loaded RAW time series data: [{name}] [{mode}] with {self.n_classes} classes.")
        print(f"Data shape: {self.data.shape}")

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)

class FewShotSampler(object):
  def __init__(self, label, sample_per_class, iterations):
    self.label            = label
    self.sample_per_class = sample_per_class
    self.all_classes      = list(set(label.tolist())) 
    self.iterations       = iterations


  def __iter__(self):
    for it in range(self.iterations):
      spc = self.sample_per_class
      batch_size = spc * len(self.all_classes) 
      few_shot_batch = []
      for i, c in enumerate(self.all_classes):
        fea_idxs = ( self.label == c).nonzero()[:,0].tolist()
        if len(fea_idxs)<spc:
            few_shot_batch.extend( random.sample(fea_idxs, max(len(fea_idxs),2)))
        else:
            few_shot_batch.extend( random.sample(fea_idxs, spc))
      batch = torch.LongTensor(few_shot_batch)
      yield batch

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations