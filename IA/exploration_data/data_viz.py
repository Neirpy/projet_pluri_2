import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IA.exploration_data.preprocessing import corr_matrix


def boxplot(df:pd.DataFrame, class_name:str):
    for col in df.columns:
        plt.boxplot(df[col], labels=df[class_name])
        plt.title(f"Boxplot of {col} for {class_name}")
        plt.xticks(rotation=90)
        plt.xlabel(class_name)
        plt.show()

def heatmap(df:pd.DataFrame, type:str):
    corr = corr_matrix(df)
    sns.heatmap(corr, annot=True)
    plt.title(f"Correlation Matrix for df {type}")
    plt.xticks(rotation=90)
    plt.show()