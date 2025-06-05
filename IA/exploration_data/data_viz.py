import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from IA.exploration_data.preprocessing import corr_matrix, pca


def boxplot(df:pd.DataFrame, class_name:str):
    """
    permet de créer des boxplot pour chaque colonne d'un dataframe
    :param df: dataframe to plot
    :class_name: class used for visualisation
    """
    categories = df[class_name].unique()
    categories.sort()
    for col in df.columns:
        if col != class_name:
            data_to_plot = [df[df[class_name] == cat][col].dropna() for cat in categories]
            plt.boxplot(data_to_plot, labels=categories)
            plt.title(f"Boxplot of {col} for {class_name}")
            plt.xticks(rotation=90)
            plt.xlabel(class_name)
            plt.show()

def heatmap(df:pd.DataFrame, type:str):
    """
    permet de créer un heatmap pour une matrice de corrélation
    :param df: df used for correlation matrix
    :param type: type of dataframe (speed or action)
    :return: None
    """
    corr = corr_matrix(df)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr, annot=True, ax=ax)
    plt.title(f"Correlation Matrix for df {type}")
    plt.xticks(rotation=90)
    plt.show()

def scatter_matrix(df:pd.DataFrame, type:str, *features):
    """
    Permet de créer une scatter matrix
    """

    components, _, explained_variance = pca(df[list(features)])

    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(explained_variance * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(4),
        color=df[type],
        title=f"Scatter Matrix for pca of {type}",
    )
    fig.update_traces(
        diagonal_visible=False,
        showlowerhalf=True,
        showupperhalf=False,
        marker=dict(opacity=0.7, size=3)
    )

    fig.show()

def elbow_method(x:pd.DataFrame):
    """
    affiche l'elbow method pour le pca
    :param x: dataframe for elbow method
    :return:None
    """
    _, _, explained_variance_ratio = pca(x)
    fig = px.area(
        x=range(1, explained_variance_ratio.shape[0] + 1),
        y=explained_variance_ratio,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.show()

def visualize_pca(df:pd.DataFrame, nb_components:int, type:str, *features):
    """
    Permet de visualiser un pca à 2 ou 3 dimensions
    :param df: dataframe to vizualise the pca
    :param nb_components: number of components for pca
    :param type: type of data for the title
    :features: list of features to visualize
    :return: None
    """
    components, loadings, _ = pca(df[list(features)], nb_components)


    if nb_components == 3:
        fig = px.scatter_3d(
            components,
            x=0,
            y=1,
            z=2,
            color=df[type],  # Use the provided series for coloring
            title=f"Visualisation for pca of {type}",
            labels={
                0 : 'PC1',
                1 : 'PC2',
                2 : 'PC3'
            }
        )
        # Add feature loading vectors as 3D lines and labels
        for i, feature in enumerate(features):
            # Add the line from origin to loading point
            fig.add_trace(
                go.Scatter3d(
                    x=[0, loadings[i, 0]],
                    y=[0, loadings[i, 1]],
                    z=[0, loadings[i, 2]],
                    mode='lines',
                    line=dict(color='red', width=4),
                    name=f'Loading: {feature}',
                    hoverinfo='name',
                    showlegend=False
                )
            )
            # Add the text label for the feature name at the loading point
            fig.add_trace(
                go.Scatter3d(
                    x=[loadings[i, 0]],
                    y=[loadings[i, 1]],
                    z=[loadings[i, 2]],
                    mode='text',
                    text=[feature],
                    textfont=dict(color='red', size=12),
                    textposition="top center",
                    hoverinfo='text',
                    showlegend=False
                )
            )

        # Customize the layout for better 3D visualization
        fig.update_layout(
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=700,
            width=900
        )

        fig.show()
    else:
        fig = px.scatter(components, x=0, y=1, color=df[type])
        for i, feature in enumerate(list(features)):
            fig.add_annotation(
                ax=0, ay=0,
                axref="x", ayref="y",
                x=loadings[i, 0],
                y=loadings[i, 1],
                showarrow=True,
                arrowsize=2,
                arrowhead=2,
                xanchor="right",
                yanchor="top"
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                yshift=5,
            )
        fig.show()