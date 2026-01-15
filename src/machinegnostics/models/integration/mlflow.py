import mlflow
from machinegnostics.models.integration import mlflow_flavor

def log_model(model, artifact_path, **kwargs):
    import mlflow.models
    return mlflow.models.Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_flavor, # Ensure this is the imported module
        model=model,
        **kwargs
    )

def autolog():
    """
    Automatically hooks into LinearRegressor.fit to log params and model.
    """
    from machinegnostics.models import LinearRegressor
    original_fit = LinearRegressor.fit

    def patched_fit(self, X, y, **kwargs):
        # 1. Log Hyper-parameters
        mlflow.log_params(self.get_params())
        
        # 2. Run actual training
        result = original_fit(self, X, y, **kwargs)
        
        # 3. Log the model artifact
        log_model(self, artifact_path="model")
        return result

    LinearRegressor.fit = patched_fit