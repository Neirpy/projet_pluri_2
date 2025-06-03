from enum import Enum
import pandas as pd
import os

class Action(Enum):
    ARRIERE = 1
    AVANT = 2
    DROITE = 3
    GAUCHE = 4
    TOURNER_DROITE = 5
    TOURNER_GAUCHE = 6
    COUCOU = 7
    NEUTRE = 8

class Vitesse(Enum):
    LENT = 1
    MOYEN = 2
    RAPIDE = 3

def concat_same_move(move:Action, *links):
    df_move = pd.DataFrame()
    id = 0
    for link in links:
        df = pd.read_csv(link)
        df['id'] = id
        df["Action"] = move.name
        df_move = pd.concat([df_move, df], ignore_index=True)
        id += 1
    return df_move

def concat_same_speed(speed:Vitesse, *links):
    df_speed = pd.DataFrame()
    id = 0
    for link in links:
        df = pd.read_csv(link)
        df['id'] = id
        df["Action"] = speed.name
        df_speed = pd.concat([df_speed, df], ignore_index=True)
        id += 1
    return df_speed

def concat_all(*dfs):
    df_all = pd.DataFrame()
    for df in dfs:
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all

def find_all_from_directory():
    current_path = os.getcwd()
    list_df_move = []
    list_df_speed = []
    df_move = pd.DataFrame()
    df_speed = pd.DataFrame()
    all_dir = os.listdir(f"IA/{current_path}/data")
    for dir in all_dir:
        name = dir.upper()
        if name == Action.name:
            df = concat_same_move(Action, *[item for item in os.listdir() if os.path.isdir(item)])
            list_df_move.append(df)
        else:
            df  = concat_same_speed(Speed, *[item for item in os.listdir() if os.path.isdir(item)])
            list_df_speed.append(df)
    df_move = concat_all(*list_df_move)
    df_speed = concat_all(*list_df_speed)
    return df_move, df_speed



