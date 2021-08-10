import os
import pandas as pd
import torch


def create_path_csv():
    df = pd.DataFrame(columns=["path", "label"])
    for v in os.walk("Real Life Violence Dataset/Violence"):
        for vid in v[2]:
            if torch.cuda.is_available():
                df = df.append({
                    "path": os.getcwd() + "/" + v[0] + "/" + vid,
                    "label": 1
                }
                    , ignore_index=True)
            else:
                df = df.append({
                    "path": os.getcwd() + "\\" + v[0].replace("/", "\\") + "\\" + vid,
                    "label": 1
                }
                    , ignore_index=True)

    for v in os.walk("Real Life Violence Dataset/NonViolence"):
        for vid in v[2]:
            if torch.cuda.is_available():
                df = df.append({
                    "path": os.getcwd() + "/" + v[0] + "/" + vid,
                    "label": 0
                }
                    , ignore_index=True)
            else:
                df = df.append({
                    "path": os.getcwd() + "\\" + v[0].replace("/", "\\") + "\\" + vid,
                    "label": 0
                }
                    , ignore_index=True)
    df.to_csv("paths.csv", index=False)
    return "paths.csv"
