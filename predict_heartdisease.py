import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads heartdisease data file into a DataFrame from a string filepath.
    """
    names=[ 'age','sex','cp','trestbps','chol','fbs','restecg', 'thalach','exang','oldpeak','slope','ca','thal','num']
    df = pd.read_csv(filepath,  names = names)
    df['heartdisease'] = df['num'].replace({0:0, 1:1, 2:1, 3:1, 4:1})
    df = df.drop('num', axis=1)
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    predictions.rename({'prediction_label': 'Heartdisease_prediction'}, axis=1, inplace=True)
    predictions['Heartdisease_prediction'].replace({1: 'Heart Disease', 0: 'No Heart Disease'},
                                            inplace=True)
    return predictions['Heartdisease_prediction']


if __name__ == "__main__":
    df = load_data('./data/heartdisease/new_cleveland.data')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)