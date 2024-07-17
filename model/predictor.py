import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from model.data import get_data, get_features


class StockPricePredictor:
    def __init__(self, ticker="^GSPC", num_years=100):
        self.ticker = ticker
        self.num_years = num_years
        self.target = "Target"
        self.model = self.create_model()
        self.data = get_data(ticker=ticker, num_years=num_years)
        self.data, self.features = get_features(data=self.data)
        self.data = self.data.dropna()
        self.threshold = 0.6

    def create_model(self, n_estimators=100, min_samples_split=100):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=1,
        )
        return model

    def train(self, data):
        self.model.fit(data[self.features], data[self.target])

    def get_predictions(self, data):
        preds = self.model.predict_proba(data[self.features])
        preds = preds[:, 1]
        preds[preds >= self.threshold] = 1
        preds[preds < self.threshold] = 0
        return preds

    def backtest(self, start=2500, step=250):
        all_tests = []
        for i in range(start, self.data.shape[0], step):
            train_data = self.data.iloc[:i].copy()
            test_data = self.data.iloc[i : (i + step)].copy()
            self.train(train_data=train_data)
            preds = self.get_predictions(data=test_data)
            preds = pd.Series(preds, index=test_data.index, name="Prediction")
            tests = pd.concat([test_data[self.target], preds], axis=1)
            all_tests.append(tests)
        all_tests = pd.concat(all_tests)
        return all_tests

    def get_precision(self):
        tests = self.backtest()
        prec = precision_score(tests["Target"], tests["Prediction"])
        return prec


def get_predictor(**kwargs):
    predictor = StockPricePredictor(**kwargs)
    predictor.train(data=predictor.data)
    return predictor


def predict_tomorrow(predictor=None, **kwargs):
    if not predictor:
        predictor = get_predictor(**kwargs)

    data = get_data(ticker=predictor.ticker, num_years=predictor.num_years)
    predict_data, _ = get_features(data=data.copy())
    predict_data = predict_data.tail(1)
    prediction = predictor.model.predict_proba(predict_data[predictor.features])[:, 1][
        0
    ]
    return prediction, data
