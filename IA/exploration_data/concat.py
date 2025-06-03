from enum import Enum
import pandas as pd
import os

class Action(Enum):
    """
    Classe contenant les différentes actions possibles pour le robot
    """
    ARRIERE = 1
    AVANT = 2
    DROITE = 3
    GAUCHE = 4
    TOURNER_DROITE = 5
    TOURNER_GAUCHE = 6
    SURPRISE = 7
    NEUTRE = 8

    def get_action(name:str):
        for action in Action:
            print(action)
            if action.name == name:
                return action
        return None


class Vitesse(Enum):
    """
    Classe contenant les différentes vitesses possibles pour le robot
    """
    LENT = 1
    MOYEN = 2
    RAPIDE = 3

    def get_vitesse(name:str):
        for vitesse in Vitesse:
            print(vitesse)
            if vitesse.name == name:
                return vitesse
        return None

def concat_same_move(move:Action, *links)->pd.DataFrame:
    """
    concatene les datasets pour un même mouvement
    :param move: une des actions de la classe Action
    :param links: une liste de liens pour les jeux de données en csv
    :return: le df concaténé pour tous les csv pour une même action, en ajoutant l'annotation et l'id
    """
    df_move = pd.DataFrame()
    id = 0
    for link in links:
        df = pd.read_csv(link)
        df['id'] = id
        df["action"] = move.name
        df_move = pd.concat([df_move, df], ignore_index=True)
        id += 1
    return df_move

def concat_same_speed(speed:Vitesse, *links)->pd.DataFrame:
    """
    concatene les datasets pour une même vitesse
    :param speed: une des vitesses de la classe Vitesse
    :param links: une liste de liens pour les jeux de données en csv
    :return: le df concaténé pour tous les csv pour une même vitesse, en ajoutant l'annotation et l'id
    """
    df_speed = pd.DataFrame()
    id = 0
    for link in links:
        df = pd.read_csv(link)
        df['id'] = id
        df["vitesse"] = speed.name
        df_speed = pd.concat([df_speed, df], ignore_index=True)
        id += 1
    return df_speed

def concat_all(*dfs)->pd.DataFrame:
    """
    Concatene tous les df donnés en entrée
    :param dfs: liste de df
    :return: df concaténé
    """
    df_all = pd.DataFrame()
    for df in dfs:
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all

def find_all_from_directory():
    """
    Cherche dans tous les dossiers du dossier data les différentes données afin de créer deux dataset : un pour la vitesse, un pour les mouvement
    sauvegarde aussi les df créés en csv dans un dossier dédié
    :return: df pour la vitesse, df pour le mouvement
    """
    current_path = os.getcwd()
    folder = os.listdir(f"{current_path}/data_regrouped_unprocessed")
    if len(folder) == 0:
        list_df_move = []
        list_df_speed = []
        df_move = pd.DataFrame()
        df_speed = pd.DataFrame()
        all_dir = os.listdir(f"{current_path}/data")
        print(all_dir)
        for folder in all_dir:
            name = folder.upper()
            print(name)
            if name in Action._member_names_:
                a = Action.get_action(name)
                df = concat_same_move(a, *[f"{current_path}/data/{folder}/{item}" for item in os.listdir(f"{current_path}/data/{folder}")])
                list_df_move.append(df)
                print(df.shape)
            elif name in Vitesse._member_names_:
                s = Vitesse.get_vitesse(name)
                df  = concat_same_speed(s, *[f"{current_path}/data/{folder}/{item}" for item in os.listdir(f"{current_path}/data/{folder}")])
                list_df_speed.append(df)
                print(df.shape)
            else:
                return df_move, df_speed
        df_move = concat_all(*list_df_move)
        df_speed = concat_all(*list_df_speed)
        df_move.to_csv(f"{current_path}/data_regrouped_unprocessed/data_move.csv")
        df_speed.to_csv(f"{current_path}/data_regrouped_unprocessed/data_speed.csv")
    else:
        df_move = pd.read_csv(f"{current_path}/data_regrouped_unprocessed/data_move.csv")
        df_speed = pd.read_csv(f"{current_path}/data_regrouped_unprocessed/data_speed.csv")
    return df_move, df_speed



