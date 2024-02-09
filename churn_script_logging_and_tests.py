'''
Logging & Tests Churn Project

Author: Samuel Haddad
Date: January, 2024
'''
import logging
import joblib
import churn_library as cls
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data - The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data - The file doesn't appear to have rows and columns")
        raise err
    return dataframe


def test_eda(perform_eda, dataframe):
    '''
    test perform eda function
    '''
    try:
        perform_eda(dataframe)
        logging.info("SUCCESS: Testing perform_eda")
    except KeyError as err:
        logging.error(
            "ERROR: Testing perform_eda - one of the variables wasn't found")
        raise err

    files = [
        'churn_distribution',
        'costumer_age_distribution',
        'marital_status_distribution',
        'total_trans_distribution',
        'heatmap']

    path_list = []

    for file in files:
        path_list.append(f'./images/eda/{file}.png')

    for file, path in zip(files, path_list):
        try:
            with open(path, encoding="utf8"):
                pass
            logging.info(
                "SUCCESS: Testing perform_eda - %s.png generated", file)
        except FileNotFoundError as err:
            logging.error(
                "ERROR: Testing perform_eda - %s.png not found", file)
            raise err


def test_encoder_helper(encoder_helper, dataframe):
    '''
    test encoder helper
    '''
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing encoder_helper - The dataframe doesn't appear to have rows and columns")
        raise err
    except NameError as err:
        logging.error(
            "ERROR: Testing encoder_helper - The dataframe wasn't found")
        raise err

    try:
        assert len(constants.cat_columns) > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing encoder_helper - The list doesn't appear to have rows and columns")
        raise err
    except AttributeError as err:
        logging.error(
            "ERROR: Testing encoder_helper - The category_lst wasn't found")
        raise err

    try:
        encoder_helper(dataframe, constants.cat_columns)
        logging.info("SUCCESS: Testing encoder_helper")
    except KeyError as err:
        logging.error(
            "ERROR: Testing encoder_helper - match problem with variables or elements of the list")
        raise err
    return dataframe


def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    test perform_feature_engineering
    '''
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering - \
                The dataframe doesn't appear to have rows and columns")
        raise err
    except NameError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering - The dataframe wasn't found")
        raise err
    try:
        _x_train, _x_test, _y_train, _y_test = perform_feature_engineering(dataframe)
        logging.info("SUCCESS: Testing perform_feature_engineering")
    except NameError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering - The dataframe wasn't found")
        raise err
    return _x_train, _x_test, _y_train, _y_test


def test_train_models(train_models, _x_train, _y_train):
    '''
    test train_models
    '''
    try:
        train_models(_x_train, _y_train)
        logging.info("SUCCESS: Testing train_models")
    except (NameError, ValueError) as err:
        logging.error(
            "ERROR: Testing train_models - At least one of the inputs wasn't found")
        raise err

    files = ['rfc_model', 'logistic_model']
    path_list = []

    for file in files:
        path_list.append(f'./models/{file}.pkl')

    for file, path in zip(files, path_list):
        try:
            with open(path, encoding="utf8"):
                pass
            logging.info(
                "SUCCESS: Testing train_models - %s.pkl generated", file)
        except FileNotFoundError as err:
            logging.error(
                "ERROR: Testing train_models - %s.pkl not found", file)
            raise err


def test_classification_report_image(
        classification_report_image,
        _x_train,
        _x_test,
        _y_train,
        _y_test):
    '''
    test classification_report_image
    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    y_train_preds_rf = rfc_model.predict(_x_train)
    y_test_preds_rf = rfc_model.predict(_x_test)
    y_train_preds_lr = lr_model.predict(_x_train)
    y_test_preds_lr = lr_model.predict(_x_test)

    try:
        classification_report_image(
            _y_train,
            _y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf)
        logging.info("SUCCESS: Testing classification_report_image")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing classification_report_image - The dataframe wasn't found")
        raise err

    files = ['logistic_results', 'rf_results', 'roc_curve_results']
    path_list = []

    for file in files:
        path_list.append(f'./images/results/{file}.png')

    for file, path in zip(files, path_list):
        try:
            with open(path, encoding="utf8"):
                pass
            logging.info(
                "SUCCESS: Testing classification_report_image - %s.png generated",
                file)
        except FileNotFoundError as err:
            logging.error(
                "ERROR: Testing classification_report_image - %s.png not found", file)
            raise err


def test_feature_importance_plot(feature_importance_plot, _x_train):
    '''
    test feature_importance_plot
    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    try:
        feature_importance_plot(
            rfc_model,
            _x_train,
            "./images/results/feature_importances.png")
        logging.info("SUCCESS: Testing feature_importance_plot")
    except NameError as err:
        logging.error(
            "ERROR: Testing feature_importance_plot - plot wasn't generated")
        raise err


if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    df = test_encoder_helper(cls.encoder_helper, df)
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df)
    test_train_models(cls.train_models, x_train, y_train)
    test_classification_report_image(
        cls.classification_report_image,
        x_train,
        x_test,
        y_train,
        y_test)
    test_feature_importance_plot(cls.feature_importance_plot, x_train)
