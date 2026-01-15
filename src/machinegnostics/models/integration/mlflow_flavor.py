import os
import pickle
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME

def save_model(model, path, mlflow_model=None, **kwargs):
    """
    Update: Added 'mlflow_model' and '**kwargs' to the signature.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # 1. Serialize the model data
    model_data_subpath = "model.pkl"
    with open(os.path.join(path, model_data_subpath), "wb") as f:
        pickle.dump(model, f)

    # 2. Use the provided mlflow_model object or create a new one
    if mlflow_model is None:
        mlflow_model = Model()
    
    # 3. Add your flavor metadata
    mlflow_model.add_flavor(
        "machinegnostics",
        model_data=model_data_subpath,
        loader_module="machinegnostics.models.integration.mlflow_flavor"
    )
    
    # 4. Add the universal python_function flavor
    mlflow_model.add_flavor(
        "python_function",
        loader_module="machinegnostics.models.integration.mlflow_flavor",
        model_data=model_data_subpath
    )

    # 5. Save the MLmodel configuration file
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

def load_model(path):
    """This function is called by MLflow to reload the model."""
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        return pickle.load(f)

def _load_pyfunc(path):
    """Required for mlflow.pyfunc.load_model support."""
    return _PyFuncModelWrapper(load_model(path))

class _PyFuncModelWrapper:
    """A wrapper to ensure predict() works with MLflow's standard format."""
    def __init__(self, model):
        self.model = model

    def predict(self, df_input):
        # MLflow predicts using DataFrames, convert to numpy for your model
        return self.model.predict(df_input.values)