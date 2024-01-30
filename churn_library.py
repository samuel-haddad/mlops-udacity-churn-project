# library doc string
'''
Churn Project

Author: Samuel Haddad
Date: January, 2024
'''

# import libraries
import os
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plot_list = [(dataframe['Churn'],
                  './images/eda/churn_distribuition.png',
                  "hist"),
                 (dataframe['Customer_Age'],
                  './images/eda/costumer_age_distribuition.png',
                  "hist"),
                 (dataframe.Marital_Status.value_counts('normalize'),
                  './images/eda//marital_status_distribuition.png',
                  "bar"),
                 (dataframe['Total_Trans_Ct'],
                  './images/eda/total_trans_distribuition.png',
                  "histplot"),
                 (dataframe[constants.quant_columns].corr(),
                  './images/eda/heatmap.png',
                  "heatmap")]

    for data, path, plot in plot_list:
        fig = plt.figure(figsize=(20, 10))
        if plot == "hist":
            data.hist()
        elif plot == "bar":
            data.plot(kind='bar')
        elif plot == "histplot":
            sns.histplot(data, stat='density', kde=True)
        elif plot == "heatmap":
            sns.heatmap(data, annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used 
                        for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        cat_list = []
        cat_groups = dataframe[['Churn', cat]].groupby(cat).mean()['Churn']

        for val in dataframe[cat]:
            cat_list.append(cat_groups.loc[val])

        dataframe[f'{cat}_Churn'] = cat_list

    return dataframe


def perform_feature_engineering(dataframe):
    '''
    input:
            dataframe: pandas dataframe

    output:
            _x_train: X training data
            _x_test: X testing data
            _y_train: y training data
            _y_test: y testing data
    '''
    y = dataframe['Churn']

    x = pd.DataFrame()
    x[constants.keep_cols] = dataframe[constants.keep_cols]

    # This cell may take up to 15-20 minutes to run
    # train test split
    _x_train, _x_test, _y_train, _y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return _x_train, _x_test, _y_train, _y_test


def train_models(_x_train, _y_train):
    '''
    train, store model results: images + scores, and store models
    input:
            _x_train: X training data
            _y_train: y training data
    output:
            None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='liblinear', max_iter=3000)

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=constants.param_grid,
        cv=5,
        error_score='raise')
    cv_rfc.fit(_x_train, _y_train)

    lrc.fit(_x_train, _y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def classification_report_image(_y_train,
                                _y_test,
                                _y_train_preds_lr,
                                _y_train_preds_rf,
                                _y_test_preds_lr,
                                _y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            _y_train: training response values
            _y_test:  test response values
            _y_train_preds_lr: training predictions from logistic regression
            _y_train_preds_rf: training predictions from random forest
            _y_test_preds_lr: test predictions from logistic regression
            _y_test_preds_rf: test predictions from random forest

    output:
            None
    '''

    # -----------#  LOGISTIC RESULTS #-----------#

    train_list = [('Logistic Regression Train', _y_train_preds_lr),
                  ('Random Forest Train', _y_train_preds_rf),
                  ]
    test_list = [('Logistic Regression Test', _y_test_preds_lr),
                 ('Random Forest Test', _y_test_preds_rf),
                 ]
    path_list = ["./images/results/logistic_results.png",
                 "./images/results/rf_results.png",
                 ]

    for train, test, pth in zip(train_list, test_list, path_list):
        fig = plt.rc('figure', figsize=(5, 5))
        plt.text(
            0.01, 1.25, str(
                train[0]), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(_y_train, train[1])), {
                 'fontsize': 10}, fontproperties='monospace')  #approach improved by OP ->monospace!
        plt.text(
            0.01, 0.6, str(
                test[0]), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(_y_test, test[1])), {
                 'fontsize': 10}, fontproperties='monospace')  #approach improved by OP ->monospace!
        plt.axis('off')

        # save fig
        plt.savefig(pth, bbox_inches='tight')
        plt.close(fig)

    # -----------#  ROC CURVE RESULTS #-----------#

    fig = plt.figure(figsize=(15, 8))

    plot_list = [('RandomForest', _y_test_preds_rf, 'blue'),
                 ('LogisticRegression', _y_test_preds_lr, 'red')
                 ]
    for graph in plot_list:
        fpr, tpr = roc_curve(_y_test, graph[1])[:2]
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=graph[2],
            label=f'{graph[0]} (AUC = %0.2f)' %
            roc_auc,
            alpha=0.8)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # save fig
    plt.savefig("./images/results/roc_curve_results.png", bbox_inches='tight')
    plt.close(fig)


def feature_importance_plot(model, _x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            _x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''

    # -----------# FEATURE IMPORTANCE #-----------#

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [_x_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(_x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(_x_data.shape[1]), names, rotation=90)

    # save fig
    plt.savefig(output_pth, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    df = encoder_helper(df, constants.cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, y_train)
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)
    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(
        rfc_model,
        X_train,
        "./images/results/feature_importances.png")
