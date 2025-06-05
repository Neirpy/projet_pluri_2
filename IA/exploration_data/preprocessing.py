import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_predict(df:pd.DataFrame) -> pd.DataFrame:
    """
    normalise les données en fonction du point de la hanche gauche
    :param df: dataframe à normaliser
    :return: dataframe normalisé
    """
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

    return new_df

def normalize(df:pd.DataFrame, label) -> pd.DataFrame:
    """
    normalise les données en fonction du point de la hanche gauche + retypage des données en float, et catégorielle pour annotation
    :param df: dataframe à normaliser
    :return: dataframe normalisé
    """
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

    new_df = new_df.iloc[:,:-1].astype(float)
    return pd.concat([new_df, df[[label]]], axis=1).reset_index(drop=True)


def pca(df, nb_components=None):
    """
    affectue un pca sur les données
    :param df: dataframe qui va subir le pca
    :param nb_components: nombre de composants de la pca
    :return: composants, loadings, et variance expliquée
    """
    if nb_components is None:
        pca = PCA()
        pca.fit(df)
    else:
        pca = PCA(n_components=nb_components)
    return pca.fit_transform(df), pca.components_.T * np.sqrt(pca.explained_variance_), np.cumsum(pca.explained_variance_ratio_)

def filter_columns(df:pd.DataFrame, *features)->pd.DataFrame:
    """
    Ne garde que les colonnes gardées en paramètre
    :param df: dataframe de base
    :param features: colonnes que l'on souhaite garder
    :return: dataframe avec uniquement les colonnes que l'on souhaite garder
    """
    new_df = df.filter(items=features)
    return new_df

def find_outliers(df):
    pass


def corr_matrix(df):
    """
    créé la matrice de corrélation en remplaçant les annotations par des valeurs numériques
    :param df: dataframe de base
    :return: matrice de corrélation
    """
    df = df.copy()
    last_col = df.columns[-1]

    le = LabelEncoder()
    df[last_col] = le.fit_transform(df[last_col])

    return df.corr()


def apply_translation(df: pd.DataFrame, exclude_columns, tx=0.01, ty=0.01):
    """
    fait une translation sur les différents points pour ajouter du bruit dans les données
    :param df: dataframe de base
    :param exclude_columns: colonnes que l'on ne veut pas modifier
    :param tx: translation pour les coordonnées x
    :param ty: translation pour les coordonnées y
    :return: dataframe avec translation
    """
    new_df = df.copy()
    for col in new_df.columns:
        if col not in exclude_columns:
            if '_X' in col:
                new_df[col] += np.random.uniform(-tx, tx)
            elif '_Y' in col:
                new_df[col] += np.random.uniform(-ty, ty)
    return new_df


def apply_zoom(df: pd.DataFrame, exclude_columns):
    """
    effectue un zoom/dezoom sur les points
    :param df: dataframe de base
    :param exclude_columns: colonnes que l'on ne souhaite pas modifier
    :return: dataframe avec zoom
    """
    new_df = df.copy()
    factor = np.random.uniform(0.95, 1.05)
    for col in new_df.columns:
        if col not in exclude_columns:
            new_df[col] = 0.5 + (new_df[col] - 0.5) * factor
    return new_df


def apply_noised(df: pd.DataFrame, exclude_columns, std=0.003):
    """
    ajoute du bruit sur les points
    :param df: dataframe de base
    :param exclude_columns: colonnes que l'on ne souhaite pas modifier
    :param std: ecart-type pour la loi normale permettant d'appliquer du bruit
    :return: dataframe bruité
    """
    new_df = df.copy()
    for col in new_df.columns:
        if col not in exclude_columns:
            new_df[col] += np.random.normal(0, std)
    return new_df


def data_augmentation(df: pd.DataFrame, exclude_columns):
    """
    ajoute des données diversifiées en se basant sur les fonctions de translation, de zoom, et de noise
    :param df: dataframe de base
    :param exclude_columns: colonnes que l'on ne souhaite pas modifier
    :return: dataframe ayant subit la data augmentation
    """
    original = df.copy()
    translated = apply_translation(df, exclude_columns)
    zoomed = apply_zoom(df, exclude_columns)
    noised = apply_noised(df, exclude_columns)

    augmented = pd.concat([original, translated, zoomed, noised], ignore_index=True)

    return augmented




