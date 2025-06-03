import pandas as pd
import seaborn as sns
import sklearn as sk


def normalize(df:pd.DataFrame):
    new_df = pd.DataFrame(columns=df.columns)
    for index, line in df.iterrows():
        x_max_ref = line.LEFT_SHOULDER_X
        y_max_ref = line.LEFT_SHOULDER_Y
        x_min_ref = line.LEFT_HIP_X
        y_min_ref = line.LEFT_HIP_Y

        dx = x_max_ref - x_min_ref
        dy = y_max_ref - y_min_ref

        scale = (dx**2 + dy**2)**0.5

        for col in df.columns:
            if "X" in col:
                val = float((line[col] - x_min_ref) / scale)
            elif "Y" in col:
                val = float((line[col] - y_min_ref) / scale)
            else:
                val = line[col]
            new_df.loc[index, col] = val
    return new_df.astype(float)


def pca(df):
    pass

def filter_columns(df:pd.DataFrame, *features)->pd.DataFrame:
    new_df = df.filter(items=features)
    return new_df

def find_outliers(df):
    pass

def corr_matrix(df):
    pass




