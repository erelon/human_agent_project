from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import os


def create_path_csv():
    df_v = pd.DataFrame(columns=["path", "label"])
    df_nv = pd.DataFrame(columns=["path", "label"])
    for v in os.walk("Real Life Violence Dataset/Violence"):
        for vid in v[2]:
            if torch.cuda.is_available():
                df_v = df_v.append({
                    "path": os.getcwd() + "/" + v[0] + "/" + vid,
                    "label": 1
                }
                    , ignore_index=True)
            else:
                df_v = df_v.append({
                    "path": os.getcwd() + "\\" + v[0].replace("/", "\\") + "\\" + vid,
                    "label": 1
                }
                    , ignore_index=True)

    for v in os.walk("Real Life Violence Dataset/NonViolence"):
        for vid in v[2]:
            if torch.cuda.is_available():
                df_nv = df_nv.append({
                    "path": os.getcwd() + "/" + v[0] + "/" + vid,
                    "label": 0
                }
                    , ignore_index=True)
            else:
                df_nv = df_nv.append({
                    "path": os.getcwd() + "\\" + v[0].replace("/", "\\") + "\\" + vid,
                    "label": 0
                }
                    , ignore_index=True)

    train_dfv, test_dfv = train_test_split(df_v, test_size=0.3)
    train_dfvn, test_dfvn = train_test_split(df_nv, test_size=0.3)

    train_df = pd.concat([train_dfv, train_dfvn])
    test_df = pd.concat([test_dfv, test_dfvn])

    train_df.to_csv("paths_train.csv", index=False)
    test_df.to_csv("paths_test.csv", index=False)
    return "paths_train.csv", "paths_test.csv"
