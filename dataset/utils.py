from collections import defaultdict
import os
import math

import random
import json
import errno
class Util:
    @staticmethod
    def mkdir_if_missing(dirname):
        """Create dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    @staticmethod
    def read_json(fpath):
        """Read json file from a path."""
        obj = None
        if fpath.split('.')[-1]=='json':
            with open(fpath, "r") as f:
                obj = json.load(f)
        elif fpath.split('.')[-1]=='jsonl':
            with open(fpath) as f:
                lines = f.readlines()
                obj = [json.loads(line) for line in lines]
        return obj




    @staticmethod
    def write_json(obj, fpath):
        """Writes to a json file."""
        Util.mkdir_if_missing(os.path.dirname(fpath))
        with open(fpath, "w") as f:
            json.dump(obj, f, indent=4, separators=(",", ": "))

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        Util.write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath):
        print(f"Reading split from {filepath}")
        split = Util.read_json(filepath)
        return split

    #read_json->read_split->generate_few_shot


