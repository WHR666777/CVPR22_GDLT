import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def zero_padding_torch(data, target_length):
    if data.shape[0] >= target_length:
        return data[:target_length]
    zero_data = torch.zeros((target_length - data.shape[0],) + data.shape[1:], dtype=data.dtype)
    return torch.cat([data, zero_data], dim=0)


def split_line(line):
    return [token for token in re.split(r"[\s,]+", line.strip()) if token]


def read_fisv_labels(label_path):
    samples = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = split_line(line)
            if len(row) < 3:
                continue
            samples.append({"video_id": row[0], "tes": float(row[1]), "pcs": float(row[2])})
    if not samples:
        raise ValueError("No valid Fis-V labels found in '{}'".format(label_path))
    return samples


def maybe_squeeze_temporal_views(data):
    if data.ndim == 3:
        return data.mean(dim=1)
    return data


class JointFisVDataset(Dataset):
    def __init__(
        self,
        visual_feat_dir,
        mm_feat_dir,
        label_path,
        clip_num=124,
        train=True,
        tes_score_max=None,
        pcs_score_max=None,
        rgb_feat_name="VST",
        flow_feat_name="I3D",
        audio_feat_name="AST",
    ):
        super().__init__()
        self.visual_feat_dir = visual_feat_dir
        self.mm_feat_dir = mm_feat_dir
        self.clip_num = clip_num
        self.train = train
        self.samples = read_fisv_labels(label_path)
        self.tes_score_max = float(tes_score_max) if tes_score_max is not None else max(sample["tes"] for sample in self.samples)
        self.pcs_score_max = float(pcs_score_max) if pcs_score_max is not None else max(sample["pcs"] for sample in self.samples)
        if self.tes_score_max <= 0 or self.pcs_score_max <= 0:
            raise ValueError("Score maxima must be positive.")

        self.rgb_data = np.load(os.path.join(mm_feat_dir, "FISV_rgb_{}.npy".format(rgb_feat_name)), allow_pickle=True).item()
        self.flow_data = np.load(os.path.join(mm_feat_dir, "FISV_flow_{}.npy".format(flow_feat_name)), allow_pickle=True).item()
        self.audio_data = np.load(os.path.join(mm_feat_dir, "FISV_audio_{}.npy".format(audio_feat_name)), allow_pickle=True).item()

    def __len__(self):
        return len(self.samples)

    def _pick_start(self, lengths):
        shared_length = min(lengths)
        if shared_length > self.clip_num:
            if self.train:
                return np.random.randint(0, shared_length - self.clip_num + 1)
            return (shared_length - self.clip_num) // 2
        return 0

    def _load_visual(self, video_id):
        feat_path = os.path.join(self.visual_feat_dir, "{}.npy".format(video_id))
        if not os.path.exists(feat_path):
            raise FileNotFoundError("Visual feature file not found: {}".format(feat_path))
        data = np.load(feat_path)
        if data.ndim != 2:
            raise ValueError("Expected 2D visual feature for '{}', got {}".format(video_id, data.shape))
        return torch.from_numpy(data).float()

    def __getitem__(self, index):
        sample = self.samples[index]
        video_id = sample["video_id"]
        if video_id not in self.rgb_data or video_id not in self.flow_data or video_id not in self.audio_data:
            raise KeyError("Missing multimodal features for '{}'".format(video_id))

        visual_seq = self._load_visual(video_id)
        rgb_seq = maybe_squeeze_temporal_views(torch.from_numpy(self.rgb_data[video_id]).float())
        flow_seq = maybe_squeeze_temporal_views(torch.from_numpy(self.flow_data[video_id]).float())
        audio_seq = maybe_squeeze_temporal_views(torch.from_numpy(self.audio_data[video_id]).float())

        start = self._pick_start([len(visual_seq), len(rgb_seq), len(flow_seq), len(audio_seq)])
        visual_seq = zero_padding_torch(visual_seq[start:start + self.clip_num], self.clip_num)
        rgb_seq = zero_padding_torch(rgb_seq[start:start + self.clip_num], self.clip_num)
        flow_seq = zero_padding_torch(flow_seq[start:start + self.clip_num], self.clip_num)
        audio_seq = zero_padding_torch(audio_seq[start:start + self.clip_num], self.clip_num)

        return {
            "visual_seq": visual_seq,
            "rgb_seq": rgb_seq,
            "flow_seq": flow_seq,
            "audio_seq": audio_seq,
            "tes_label": torch.tensor(sample["tes"] / self.tes_score_max, dtype=torch.float32),
            "pcs_label": torch.tensor(sample["pcs"] / self.pcs_score_max, dtype=torch.float32),
            "video_id": video_id,
        }


def build_joint_datasets(
    visual_feat_dir,
    mm_feat_dir,
    train_label_path,
    test_label_path,
    clip_num,
    tes_score_max=None,
    pcs_score_max=None,
    rgb_feat_name="VST",
    flow_feat_name="I3D",
    audio_feat_name="AST",
):
    train_dataset = JointFisVDataset(
        visual_feat_dir=visual_feat_dir,
        mm_feat_dir=mm_feat_dir,
        label_path=train_label_path,
        clip_num=clip_num,
        train=True,
        tes_score_max=tes_score_max,
        pcs_score_max=pcs_score_max,
        rgb_feat_name=rgb_feat_name,
        flow_feat_name=flow_feat_name,
        audio_feat_name=audio_feat_name,
    )
    test_dataset = JointFisVDataset(
        visual_feat_dir=visual_feat_dir,
        mm_feat_dir=mm_feat_dir,
        label_path=test_label_path,
        clip_num=clip_num,
        train=False,
        tes_score_max=train_dataset.tes_score_max,
        pcs_score_max=train_dataset.pcs_score_max,
        rgb_feat_name=rgb_feat_name,
        flow_feat_name=flow_feat_name,
        audio_feat_name=audio_feat_name,
    )
    return train_dataset, test_dataset
