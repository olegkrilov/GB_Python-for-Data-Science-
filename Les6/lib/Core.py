import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Constants
DATA = 'data'
TARGET = 'target'
FEAT = 'feature_names'
PRICE = 'price'

CORRECT_VALUE = 'Correct'
PREDICTED_VALUE = 'Precited'
ERROR = 'Error'

TEST_RANGE = .3
RANDOM_SEED = 42


# Get Data
def load_data():
    Source = load_boston()
    return pd.DataFrame(Source[DATA], columns=Source[FEAT]), pd.DataFrame(Source[TARGET], columns=[PRICE])
    

# Main solution
def train_model(
    train_data, 
    train_answers, 
    test_data, 
    test_answers, 
    model = LinearRegression(),
    score_fn = r2_score
):
    model.fit(train_data, train_answers)
    predicted_values = model.predict(test_data).flatten()
    
    results = pd.DataFrame({
        CORRECT_VALUE: test_answers,
        PREDICTED_VALUE: predicted_values
    })
    
    results[ERROR] = results[PREDICTED_VALUE] - results[CORRECT_VALUE]
    
    return results, score_fn(results[PREDICTED_VALUE], results[CORRECT_VALUE])

        