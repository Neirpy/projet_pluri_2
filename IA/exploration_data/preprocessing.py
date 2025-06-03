import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

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


def pca(df, nb_components=None):
    if nb_components is None:
        pca = PCA()
        pca.fit(df)
        return np.cumsum(pca.explained_variance_ratio_)
    else:
        pca = PCA(n_components=nb_components)
        components = pca.fit_transform(df)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        return components, loadings

def filter_columns(df:pd.DataFrame, *features)->pd.DataFrame:
    new_df = df.filter(items=features)
    return new_df

def find_outliers(df):
    pass

def corr_matrix(df):
    return df.corr()

def data_augmentation(df:pd.DataFrame):
    pass




