from torch.utils.data import Dataset
import numpy as np
import os
import torch
import re


class AQADataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26, action_type='all',
                 score_key='Total_Score', score_max=None, dataset='rg', train=True):
        self.train = train
        self.dataset = dataset.lower()
        self.video_path = video_feat_path
        self.clip_num = clip_num
        self.score_key = score_key

        self.labels = self.read_label(label_path, score_key, action_type)
        if len(self.labels) == 0:
            raise ValueError(
                "No samples were loaded from '{}' with dataset='{}', action_type='{}', score_key='{}'".format(
                    label_path, self.dataset, action_type, score_key
                )
            )

        if score_max is None:
            score_max = self.default_score_max()
        self.score_max = float(score_max)
        if self.score_max <= 0:
            raise ValueError("score_max must be positive, got {}".format(self.score_max))

    def default_score_max(self):
        if self.dataset == 'rg':
            return 25.0
        return max(score for _, score in self.labels)

    def read_label(self, label_path, score_key, action_type):
        with open(label_path, 'r', encoding='utf-8') as fr:
            rows = [self.split_line(line) for line in fr if line.strip()]
        if not rows:
            raise ValueError("Label file '{}' is empty".format(label_path))

        has_header = self.has_header(rows[0], score_key)
        header = rows[0] if has_header else None
        data_rows = rows[1:] if has_header else rows

        if header is not None:
            header_map = {name: idx for idx, name in enumerate(header)}
            video_idx = self.find_video_index(header_map)
            score_idx = self.find_score_index(header_map, score_key)
        else:
            video_idx = 0
            score_idx = self.legacy_score_index(score_key)

        labels = []
        for row in data_rows:
            if len(row) <= max(video_idx, score_idx):
                continue
            video_id = row[video_idx]
            if self.dataset == 'rg' and action_type != 'all' and action_type != video_id.split('_')[0]:
                continue
            labels.append([video_id, float(row[score_idx])])
        return labels

    @staticmethod
    def split_line(line):
        return [token for token in re.split(r'[\s,]+', line.strip()) if token]

    @staticmethod
    def has_header(row, score_key):
        known_keys = {'difficulty_score', 'execution_score', 'total_score', 'tes', 'pcs', 'video', 'video_id', 'id'}
        lowered = {token.lower() for token in row}
        return score_key.lower() in lowered or bool(lowered & known_keys)

    @staticmethod
    def find_video_index(header_map):
        lowered_map = {key.lower(): value for key, value in header_map.items()}
        for key in ('video_id', 'video', 'id', 'name', 'filename'):
            if key in lowered_map:
                return lowered_map[key]
        return 0

    @staticmethod
    def find_score_index(header_map, score_key):
        lowered_map = {key.lower(): value for key, value in header_map.items()}
        if score_key.lower() not in lowered_map:
            raise KeyError("Could not find score column '{}' in label header {}".format(score_key, list(header_map.keys())))
        return lowered_map[score_key.lower()]

    @staticmethod
    def legacy_score_index(score_key):
        idx = {
            'Difficulty_Score': 1,
            'Execution_Score': 2,
            'Total_Score': 3,
            'TES': 1,
            'PCS': 2,
        }
        if score_key not in idx:
            raise KeyError("Unknown score_key '{}' for headerless label file".format(score_key))
        return idx[score_key]

    def __getitem__(self, idx):
        feat_path = os.path.join(self.video_path, self.labels[idx][0] + '.npy')
        if not os.path.exists(feat_path):
            raise FileNotFoundError("Feature file not found: {}".format(feat_path))

        video_feat = np.load(feat_path)
        if video_feat.ndim != 2:
            raise ValueError("Expected feature shape [T, C], got {} from '{}'".format(video_feat.shape, feat_path))

        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num + 1)
                video_feat = video_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]), dtype=video_feat.dtype)
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat

        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_max
