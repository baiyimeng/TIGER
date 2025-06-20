import os
import json
import numpy as np
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):
    def __init__(self, data_path, dataset, max_his_len, mode, padding_pos="right"):
        super().__init__()
        self.max_his_len = max_his_len
        self.mode = mode
        self.padding_pos = padding_pos

        file_path = os.path.join(data_path, dataset, f"{dataset}.inter.json")
        with open(file_path, "r") as f:
            self.inters = json.load(f)

        self.item_num = max((max(v) for v in self.inters.values()), default=0) + 1

        if mode == "train":
            self.inter_data = self._build_data(mode="train")
        elif mode == "valid":
            self.inter_data = self._build_data(mode="valid")
        elif mode == "test":
            self.inter_data = self._build_data(mode="test")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        return self.inter_data[index]

    def _pad_sequence(self, seq):
        if len(seq) < self.max_his_len:
            pad_len = self.max_his_len - len(seq)
            padding = [0] * pad_len
            return seq + padding if self.padding_pos == "right" else padding + seq
        else:
            return seq[-self.max_his_len :]

    def _build_data(self, mode):
        inter_data = []
        for uid, items in self.inters.items():
            items = [i + 1 for i in items]  # +1 to reserve 0 for padding
            uid = int(uid)

            if mode == "train":
                item_seq = items[:-2]
                for i in range(1, len(item_seq)):
                    his = self._pad_sequence(item_seq[:i])
                    inter_data.append(
                        {
                            "his": np.array(his, dtype=np.int64),
                            "tgt": np.array(item_seq[i], dtype=np.int64),
                            "user": np.array(uid, dtype=np.int64),
                        }
                    )

            elif mode == "valid":
                if len(items) < 2:
                    continue  # skip invalid short sequences
                his = self._pad_sequence(items[:-2])
                inter_data.append(
                    {
                        "his": np.array(his, dtype=np.int64),
                        "tgt": np.array(items[-2], dtype=np.int64),
                        "user": np.array(uid, dtype=np.int64),
                    }
                )

            elif mode == "test":
                if len(items) < 2:
                    continue
                his = self._pad_sequence(items[:-1])
                inter_data.append(
                    {
                        "his": np.array(his, dtype=np.int64),
                        "tgt": np.array(items[-1], dtype=np.int64),
                        "user": np.array(uid, dtype=np.int64),
                    }
                )

        return inter_data
