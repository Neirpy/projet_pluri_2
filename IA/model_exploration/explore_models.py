from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

import numpy as np
import matplotlib.pyplot as plt


def explore_models(X, y):
    """
    Explore différents modèles de machine learning sur les données X et y.
    Affiche les résultats des modèles et les scores moyens et l'écart-type.
    Retourne les résultats des 3 meilleurs modèles et le premier modèle.
    """
    results = train_models(X, y)

    print("=== Moyenne et écart-type des scores ===")
    summary = {
        name: {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        for name, scores in results.items()
    }

    for name, stats in summary.items():
        print(f"{name}: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")

    print("\n=== Top 3 modèles (détail des scores) ===")
    top_3 = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    is_first_model = True
    first_model = None
    for name, stats in top_3:
        print(f"{name}: scores = {stats['scores']}")
        print(f"\nModèle: {name}")

        if name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif name == 'Gradient Boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif name == 'SVM Linear':
            model = SVC(kernel='linear', random_state=42)
        elif name == 'SVM RBF':
            model = SVC(kernel='rbf', random_state=42)
        elif name == 'SVM Poly':
            model = SVC(kernel='poly', random_state=42)
        elif name == 'SVM Sigmoid':
            model = SVC(kernel='sigmoid', random_state=42)
        elif name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
        elif name == 'KNN Cosine':
            model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        elif name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif name == 'SGD Classifier':
            model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        elif name == 'ANN':
            model = KerasClassifier(
                model=create_ann_model,  # ✅ remplace build_fn
                input_dim=X.shape[1],
                nb_outputs=len(np.unique(y)),
                nb_layers=2,
                nb_neurons=8,
                epochs=50,
                batch_size=100,
                verbose=0
            )
        else:
            print("Modèle non reconnu pour la matrice de confusion.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        if is_first_model:
            is_first_model = False
            first_model = model


        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Matrice de confusion - {name}')
        plt.show()

    if 'Random Forest' in results:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n=== Random Forest Feature Importances ===")
        for f in range(min(5, X.shape[1])):
            print(f"{f + 1}. Feature {indices[f]}: {importances[indices[f]]:.4f}")

    if 'Gradient Boosting' in results:
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X, y)
        importances = gb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n=== Gradient Boosting Feature Importances ===")
        for f in range(min(5, X.shape[1])):
            print(f"{f + 1}. Feature {indices[f]}: {importances[indices[f]]:.4f}")

    return top_3, first_model

# Toutes les fonctions d'exploration des modèles
def explore_random_forest(X, y):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

    return rf_scores

def explore_gradient_boosting(X, y):
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_scores = cross_val_score(gb_model, X, y, cv=5, scoring='accuracy')

    return gb_scores

def explore_svm_linear(X, y):
    svm_model = SVC(kernel='linear', random_state=42)
    svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

    return svm_scores

def explore_svm_rbf(X, y):
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

    return svm_scores

def explore_svm_poly(X, y):
    svm_model = SVC(kernel='poly', random_state=42)
    svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

    return svm_scores

def explore_svm_sigmoid(X, y):
    svm_model = SVC(kernel='sigmoid', random_state=42)
    svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

    return svm_scores

def explore_knn(X, y):
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

    return knn_scores

def explore_knn_cosine(X, y):
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

    return knn_scores

def explore_logistic_regression(X, y):
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')

    return lr_scores

def explore_sgd_classifier(X, y):
    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_scores = cross_val_score(sgd_model, X, y, cv=5, scoring='accuracy')

    return sgd_scores

def explore_ann(X, y, nb_layers=3, nb_neurons=32):
    input_dim = X.shape[1]
    nb_outputs = len(np.unique(y))

    ann_model = KerasClassifier(
        model=create_ann_model,
        input_dim=input_dim,
        nb_outputs=nb_outputs,
        nb_layers=nb_layers,
        nb_neurons=nb_neurons,
        epochs=50,
        batch_size=100,
        verbose=0
    )

    ann_scores = cross_val_score(ann_model, X, y, cv=5, scoring='accuracy')
    return ann_scores

def create_ann_model(input_dim, nb_outputs, nb_layers=2, nb_neurons=8):
    """
    Crée un modèle de réseau de neurones artificiels (ANN) avec Keras.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(nb_neurons, activation='relu'))
    model.add(Dropout(0.2))
    for _ in range(nb_layers - 1):
        model.add(Dense(nb_neurons, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(nb_outputs, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(X, y):
    """
    Entraîne plusieurs modèles de machine learning sur les données X et y.
    """
    models_results = {}

    print("Training Random Forest...")
    models_results['Random Forest'] = explore_random_forest(X, y)
    print("-> Random Forest result:", models_results['Random Forest'])

    print("Training Gradient Boosting...")
    models_results['Gradient Boosting'] = explore_gradient_boosting(X, y)
    print("-> Gradient Boosting result:", models_results['Gradient Boosting'])

    print("Training SVM Linear...")
    models_results['SVM Linear'] = explore_svm_linear(X, y)
    print("-> SVM Linear result:", models_results['SVM Linear'])

    print("Training SVM RBF...")
    models_results['SVM RBF'] = explore_svm_rbf(X, y)
    print("-> SVM RBF result:", models_results['SVM RBF'])

    print("Training SVM Poly...")
    models_results['SVM Poly'] = explore_svm_poly(X, y)
    print("-> SVM Poly result:", models_results['SVM Poly'])

    print("Training SVM Sigmoid...")
    models_results['SVM Sigmoid'] = explore_svm_sigmoid(X, y)
    print("-> SVM Sigmoid result:", models_results['SVM Sigmoid'])

    print("Training KNN...")
    models_results['KNN'] = explore_knn(X, y)
    print("-> KNN result:", models_results['KNN'])

    print("Training KNN Cosine...")
    models_results['KNN Cosine'] = explore_knn_cosine(X, y)
    print("-> KNN Cosine result:", models_results['KNN Cosine'])

    print("Training Logistic Regression...")
    models_results['Logistic Regression'] = explore_logistic_regression(X, y)
    print("-> Logistic Regression result:", models_results['Logistic Regression'])

    print("Training SGD Classifier...")
    models_results['SGD Classifier'] = explore_sgd_classifier(X, y)
    print("-> SGD Classifier result:", models_results['SGD Classifier'])

    print("Training ANN...")
    models_results['ANN'] = explore_ann(X, y, nb_layers=2, nb_neurons=8)
    print("-> ANN result:", models_results['ANN'])

    return models_results
