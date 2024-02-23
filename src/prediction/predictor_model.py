import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from xgboost import XGBRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(task_name="model")
PREDICTOR_FILE_NAME = "predictor.joblib"

PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the XGBoost Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "XGBoost Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = 6,
        max_leaves: Optional[int] = 0,
        gamma: Optional[float] = 0.0,
        learning_rate: Optional[float] = 0.1,
        lags: Union[int, List[int]] = 20,
        use_exogenous: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new XGBoost Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = forecast horizon * lags_forecast_ratio
                This parameters overides lags parameters.

            n_estimators (int): The maximum number of estimators at which boosting is terminated. In case of perfect fit,
                the learning procedure is stopped early. Values must be in the range [1, inf).


            max_depth (Optional[int]): Maximum tree depth for base learners

            max_leaves (Optional[int]): Maximum number of leaves; 0 indicates no limit.

            gamma (Optional[float]): - (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.

            learning_rate (Optional[float]): Boosting learning rate (xgb's “eta”)

            lags (Union[int, List[int]]): Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                - int: include lags from 1 to lags (included).
                - list, 1d numpy ndarray or range: include only lags present in lags, all elements must be int.

            use_exogenous (bool):
                If true, uses covariates in training.

            random_state (int): Sets the underlying random seed at model initialization time.

            kwargs (dict): Additional parameters accepted by the sklearn base model.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.gamma = gamma
        self.random_state = random_state
        self.lags = lags
        self.use_exogenous = use_exogenous
        self._is_trained = False
        self.models = {}
        self.data_schema = data_schema
        self.end_index = {}
        self.history_length = None

        has_covariates = len(
            data_schema.future_covariates + data_schema.static_covariates
        ) > 0 or data_schema.time_col_dtype in ["DATE", "DATETIME"]
        self.use_exogenous = use_exogenous and has_covariates

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )
        if lags_forecast_ratio:
            lags = self.data_schema.forecast_length * lags_forecast_ratio
            self.lags = lags

        self.base_model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            gamma=self.gamma,
            random_state=self.random_state,
            **kwargs,
        )

        self.transformer_exog = MinMaxScaler() if has_covariates else None

    def _add_future_covariates_from_date(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """
        Creates future covariates from the data column.

        Args:
            history (pd.DataFrame):
                The data to create the covariates on.

            data_schema (ForecastingSchema):
                Schema of the data.

            is_training (bool):
                Set to true if the process is to be done on the training data.

        Returns (pd.DataFrame): The processed dataframe
        """

        future_covariates_names = data_schema.future_covariates
        if data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            date_col = pd.to_datetime(history[data_schema.time_col])
            year_col = date_col.dt.year
            month_col = date_col.dt.month
            year_col_name = f"{data_schema.time_col}_year"
            month_col_name = f"{data_schema.time_col}_month"
            history[year_col_name] = year_col
            history[month_col_name] = month_col
            if is_training:
                future_covariates_names += [year_col_name, month_col_name]

        return history

    def crop_data(self, all_series: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Reduces the data based on the history_length attribute.

        Args:
            all_series (List[pd.DataFrame]): List of the original data to be reduced.

        Returns (List[pd.DataFrame]): List of the reduced data.

        """

        new_length = []
        for series in all_series:
            series = series.iloc[-self.history_length :]
            new_length.append(series.copy())
        all_series = new_length
        return all_series

    def _validate_lags_and_history_length(self, series_length: int):
        """
        Validate the value of lags and that history length is at least double the forecast horizon.
        If the provided lags value is invalid (too large), lags are set to the largest possible value.

        Args:
            series_length (int): The length of the history.

        Returns: None
        """
        if series_length < 2 * self.data_schema.forecast_length:
            raise ValueError(
                f"Training series is too short. History should be at least double the forecast horizon. history_length = ({series_length}), forecast horizon = ({self.data_schema.forecast_length})"
            )

        if self.lags >= series_length:
            logger.warning(
                f"The maximum lag ({self.lags}) must be less than the length of the series ({series_length}). Lags set to ({series_length - 1})"
            )
            self.lags = series_length - 1

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate Adaboost model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
        """
        np.random.seed(self.random_state)
        data_schema = self.data_schema
        history = self._add_future_covariates_from_date(
            history=history, data_schema=data_schema, is_training=True
        )
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col).reset_index()
            for id_ in all_ids
        ]

        if self.history_length:
            all_series = self.crop_data(all_series)

        targets = [series[data_schema.target] for series in all_series]
        target_series = pd.DataFrame({f"id_{k}": v for k, v in zip(all_ids, targets)})

        self._validate_lags_and_history_length(series_length=len(targets[0]))

        exog = None

        if self.use_exogenous:
            covariates_names = (
                data_schema.future_covariates + data_schema.static_covariates
            )
            exog = [series[covariates_names] for series in all_series]
            exog = pd.concat(exog, axis=1)
            exog.columns = [str(i) for i in range(exog.shape[1])]
            self.train_end_index = all_series[0].index.values[-1]

        self.model = ForecasterAutoregMultiSeries(
            regressor=self.base_model,
            lags=self.lags,
            transformer_series=MinMaxScaler(),
            transformer_exog=self.transformer_exog,
        )

        self.model.fit(series=target_series, exog=exog)

        self.all_ids = all_ids
        self._is_trained = True

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        test_data = self._add_future_covariates_from_date(
            history=test_data, data_schema=self.data_schema, is_training=False
        )

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_)
            .drop(columns=self.data_schema.id_col)
            .reset_index()
            for id_ in self.all_ids
        ]
        exog = None
        if self.use_exogenous:
            covariates_names = (
                self.data_schema.future_covariates + self.data_schema.static_covariates
            )
            exog = [series[covariates_names] for series in all_series]
            exog = pd.concat(exog, axis=1)
            exog.columns = [str(i) for i in range(exog.shape[1])]
            start = self.train_end_index + 1
            exog.index = pd.RangeIndex(
                start=start,
                stop=start + self.data_schema.forecast_length,
            )

        forecast = self.model.predict(steps=self.data_schema.forecast_length, exog=exog)
        forecast.columns = [c.split("id_")[1] for c in forecast.columns]
        predictions = []
        for column in forecast.columns:
            predictions += forecast[column].values.tolist()

        test_data[prediction_col_name] = predictions

        return test_data

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(history=history)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
