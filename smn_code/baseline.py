import time
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from typing import List

nltk.download('stopwords')


class Normalizer(BaseEstimator, TransformerMixin):
    """
    Class for normalizing texts. Normalization means turning all letters to lowercase, removing trailing spaces and
    stopwords, and performing stemming.
    """

    def __init__(self):
        print('_init_ is called')

    def fit(self, x=None, y=None):
        print('fit is called')
        return self

    def transform(self, x, y=None):
        print('transform is called')

        x_transformed = x.copy()

        # Remove uppercase letters
        x_transformed = x_transformed.apply(lambda x: ' '.join(x.lower() for x in str(x).split()))

        # Remove non-alpha characters (the norwegian letters æ, ø and å are kept)
        x_transformed = x_transformed.apply(lambda x: ' '.join([re.sub('[^a-zæøå]+', '', x) for x in str(x).split()]))

        # If there are several spaces after each other, remove the extra ones
        x_transformed = x_transformed.apply(lambda x: re.sub(' +', ' ', x))

        # Remove stopwords
        stop = nltk.corpus.stopwords.words('norwegian')
        x_transformed = x_transformed.apply(lambda x: ' '.join([x for x in x.split() if x not in stop]))

        # Perform stemming
        stemmer = nltk.stem.snowball.NorwegianStemmer()
        x_transformed = x_transformed.apply(lambda x: ' '.join([stemmer.stem(x) for x in x.split()]))

        return x_transformed


def grid_search(classifier: object, params: dict, train_vectorized, y_train) -> None:
    """
    Function to perform grid search on a model.
    Args:
        classifier: the model to tune, such as SVC() etc.
        params: the parameters to tune, and values to try. Format: 'model_name__parameter_name': [val1, val2,...]
        train_vectorized: the training data to use, should be vectorized
        y_train: list of y-labels
    Returns:
        None
    """

    model_pipe = Pipeline([
        ('clf', classifier)
    ])

    search = GridSearchCV(model_pipe, params, verbose=1)
    search.fit(train_vectorized, y_train)
    print(f"Classifier is: {str(classifier)}")
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)


def find_best_params(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Function to perform a gridsearch for the best hyperparameters for three ML models: SVM, KNN and logistic regression. 
    It will print the best results. 
    Args:
        train: dataset to train the models on the format |text|label|
        test: dataset to test the models on the format |text|label|
    returns:
        None
    """

    X_train = train['text']
    y_train = train['label']
    X_test = test['text']
    y_test = test['label']

    # Perform a parameter search for different models

    # First vectorize the data so that it will not have to be vectorized for each attempted set of parameters
    # later. 
    feature_pipe = Pipeline([
        ('normalization', Normalizer()),
        ('feature_extraction', TfidfVectorizer()),
    ])

    vectorized_train = feature_pipe.fit_transform(X_train)

    param_svc = {
        'clf__C': [0.01, 0.1, 1, 10, 100, 1000],  # C increases -> fewer mistakes and more complex margin
        'clf__gamma': [0.01, 0.1, 1, 10, 100, 1000],  # gamma increases -> each examples influence gets shorter
    }

    param_logistic_regression = {
        'clf__solver': ['lbfgs', 'newton-cg', 'sag'],
        'clf__penalty': ['l2', 'none'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],  # C increases -> fewer mistakes
    }

    param_knn = {
        'clf__leaf_size': [5, 10, 30, 50],
        'clf__n_neighbors': [5, 10, 20],
        'clf__weights': ['uniform', 'distance'],
    }

    grid_search(SVC(), param_svc, vectorized_train, y_train)
    grid_search(LogisticRegression(max_iter=2000), param_logistic_regression, vectorized_train, y_train)
    grid_search(KNeighborsClassifier(), param_knn, vectorized_train, y_train)


def test_best_models(train: pd.DataFrame, test: pd.DataFrame, classifiers: List) -> None:
    """
    Use the various models to perform prediction and print the results. The parameters in the classifiers below were
    found and printed by find_best_params. A confusion matrix visualizing the results is also created and saved to file. 
    Args:
        train: training dataset on the format |text|label|
        test: test dataset on the format |text|label|
        classifiers: a list of classifiers to use for testing/evaluation. E.g. [SVC(C=1000, gamma=0.01), LogisticRegression(max_iter=2000,
            C=100, penalty=None, solver='sag')] would be valid.
    returns:
        None
    """
    
    X_train = train['text']
    y_train = train['label']
    X_test = test['text']
    y_test = list(test['label'])

    # First vectorize the data
    feature_pipe = Pipeline([
        ('normalization', Normalizer()),
        ('feature_extraction', TfidfVectorizer()),
    ])

    vectorized_train = feature_pipe.fit_transform(X_train)
    vectorized_test = feature_pipe.transform(X_test)

    # Use all classifiers to make predictions and see which one performs the best
    for classifier in classifiers:
        clf_pipe = Pipeline([
            ('clf', classifier)
        ])

        clf_pipe.fit(vectorized_train, y_train)

        # Make predictions and check prediction time
        start_time = time.time()
        y_pred = clf_pipe.predict(vectorized_test)
        print(f'{str(classifier)} uses {time.time() - start_time} seconds on predictions')

        # Check evaluation metrics
        print(f'Using {str(classifier)} as a classifier gave the following results:')
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        confusion_matrix_results = confusion_matrix(y_test, y_pred, normalize='true')
        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_results,
                                            display_labels=['Negative', 'Positive'])
        cm_display = cm_display.plot(cmap=plt.cm.RdPu)
        cm_display.figure_.savefig(f'./figures/confusion_matrix_{classifier.__class__.__name__}.png',
                                   dpi='figure')
        plt.show()
        cm_display.figure_.clf()