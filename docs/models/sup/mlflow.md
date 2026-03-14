# MLflow Integration with Machine Gnostics

This guide provides a comprehensive workflow for tracking experiments, versioning, and deploying **Machine Gnostics** models using [MLflow](https://mlflow.org/).

## Overview

Machine Gnostics integrates seamlessly with MLflow to provide industrial-grade MLOps capabilities for your robust models. This integration allows you to:

*   **Experiment Tracking:** Log hyperparameters, gnostic metrics (like R-Entropy), and model artifacts automatically.
*   **Model Versioning:** Manage different versions of your gnostic models in a centralized registry.
*   **Reproducibility:** Capture the exact environment, data, and parameters used for training.
*   **Deployment:** Serve models using standard deployment patterns via the `pyfunc` interface (Docker, generic APIs).

---

## 1. Installation & Setup

Ensure you have both libraries installed in your environment:

```bash
pip install machinegnostics mlflow
```

### Initial Configuration

To configure MLflow in your Python script or notebook, set the tracking URI and experiment name.

```python
import mlflow
from machinegnostics.integration import mlflow as mg_mlflow

# 1. Set the tracking URI (where data is stored)
# Default is a local folder called ./mlruns
mlflow.set_tracking_uri("./mlruns")

# 2. Set the experiment name
# Best practice: Name this after your specific project or problem
mlflow.set_experiment("Machine_Gnostics_Experiment")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
```

---

## 2. Basic Model Logging

The core workflow involves training a model inside an `mlflow.start_run()` context block. This ensures all metrics and artifacts are tied to a specific "run".

```python
import numpy as np
from mlflow.models import infer_signature
from machinegnostics.models import LogisticRegressor

# Sample Data
X_train = np.random.rand(100, 2)
y_train = np.random.randint(0, 2, 100)

# Start the Run
with mlflow.start_run(run_name='Gnostic_LogReg_v1') as run:
    
    # 1. Initialize and Train
    model = LogisticRegressor(degree=2, gnostic_characteristics=True)
    model.fit(X_train, y_train)
    
    # 2. Log Parameters (Hyperparameters)
    mlflow.log_param("degree", model.degree)
    mlflow.log_param("gnostic_characteristics", True)
    mlflow.log_param("model_type", "LogisticRegressor")
    
    # 3. Log Metrics (Performance)
    # R-Entropy (re) is a specific internal metric for Machine Gnostics
    mlflow.log_metric("rentropy", model.re) 
    
    # Optional: Log training iterations if available
    if hasattr(model, '_history') and isinstance(model._history, list):
         mlflow.log_metric("iterations", len(model._history))

    # 4. Create Signature (Describes Input/Output Schema)
    # This helps MLflow validate input data types during serving
    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)
    
    # 5. Log the Model
    # Uses machinegnostics custom integration to package the model correctly
    mg_mlflow.log_model(
        model, 
        artifact_path="model",
        signature=signature,
        input_example=X_train[:3]
    )
    
    print(f"Run ID: {run.info.run_id}")
```

---

## 3. Loading Models

There are two primary ways to load a saved model, depending on whether you are analyzing the model interactively or deploying it to production.

### Method A: Load as PyFunc (Production)

Use this standard method if you only need the `predict()` function. This format is generic and works with deployment tools like Docker, AWS SageMaker, or Azure ML.

```python
import pandas as pd

# Replace with your actual Run ID
run_id = "YOUR_FITTED_RUN_ID"
logged_model_uri = f"runs:/{run_id}/model"

# Load generic model wrapper
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Predict 
# MLflow PyFunc generally expects DataFrame inputs for compatibility
# It ensures columns match the signature provided during logging
test_df = pd.DataFrame(X_train[:5])
predictions = loaded_model.predict(test_df)

print(predictions)
```

### Method B: Load Native Model (Research)

Use this method if you need to inspect internal attributes like `weights`, `history`, or `re` (R-Entropy) that are specific to the Machine Gnostics object.

```python
from machinegnostics.integration.mlflow_flavor import load_model

# Download artifacts locally
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, 
    artifact_path="model"
)

# Load original Machine Gnostics object instance
native_model = load_model(local_path)

# Now you have full access to the class attributes
print(f"Model Degree: {native_model.degree}")
print(f"Model Weights: {native_model.weights}")
print(f"R-Entropy: {native_model.re}")
```

---

## 4. Comparing Experiments in the UI

You can visualize your runs to compare hyperparameters and performance metrics.

1.  Open your terminal in the directory where you ran the script.
2.  Run the command:
    ```bash
    mlflow ui
    ```
3.  Open your browser to [http://localhost:5000](http://localhost:5000).

**What to look for:**

*   **Metrics:** Compare `rentropy` (Residual Entropy) or `accuracy` across different runs to find the most robust model.
*   **Parameters:** See which combinations of `degree` or regularization settings produced the best results.
*   **Artifacts:** Download the saved model files or custom plots manually if needed.

---

## 5. Model Registry & Deployment

The Model Registry acts as a centralized store for managing the lifecycle of your models (e.g., Staging vs. Production).

### Step 1: Register a Model

You can register a model programmatically from an existing run, assigning it a unique name.

```python
model_name = "Production_Gnostic_Classifier"
run_id = "BEST_RUN_ID_FROM_COMPARISON"

# Register the model
registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

print(f"Version: {registered_model.version}")
```

### Step 2: Manage Stages

Move models through lifecycle stages using the `MlflowClient`. For example, promoting a model to 'Production'.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition specific version to Production
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)
```

### Step 3: Load Production Model

In your production application (e.g., FastAPI service), load the model using its stage alias. This allows you to update the underlying model version without minimizing downtime or changing your application code.

```python
# Load specifically from the "Production" alias
model_uri = f"models:/{model_name}/Production"
production_model = mlflow.pyfunc.load_model(model_uri)

# Ready to serve
result = production_model.predict(new_data)
```

---

## 6. Advanced Features

### Logging Custom Artifacts

You can save plots, data arrays, or text files alongside your model for better documentation and auditability.

```python
import matplotlib.pyplot as plt

with mlflow.start_run():
    # ... training code ...
    
    # 1. Save a plot locally
    plt.plot(model._history) # Assuming history is suitable for plotting
    plt.title("Optimization Trajectory")
    plt.savefig("optimization_curve.png")
    
    # 2. Log the image file to MLflow
    mlflow.log_artifact("optimization_curve.png")
    
    # 3. Log raw weights for audit
    np.save('weights.npy', model.weights)
    mlflow.log_artifact('weights.npy')
```

### Tags and Notes

Adding metadata helps organize large projects and provides context for team members.

```python
with mlflow.start_run():
    # Set searchable key-value tags
    mlflow.set_tag("dataset", "sensor_data_v2")
    mlflow.set_tag("developer", "Nirmal")
    mlflow.set_tag("gnostic_type", "RobustRegression")
    mlflow.set_tag("production_ready", "yes")
    
    # Add a comprehensive description note
    mlflow.set_tag("mlflow.note.content", 
                   "This model uses the new gnostic weighting algorithm to handle "
                   "outliers in the sensor stream. Trained on 200 samples.")
```

---

## Best Practices Summary

1.  **Always use `infer_signature`:** This ensures your model deployment knows exactly what input shape and types to expect.
2.  **Log `input_example`:** This stores a snippet of your data with the model, making it easier for consumers to understand the required input format.
3.  **Track R-Entropy:** Since Machine Gnostics focuses on minimizing gnostic entropy/information, always log `model.re` as a key metric.
4.  **Use Context Managers:** Always use `with mlflow.start_run():` to ensure runs are closed properly, even if exceptions occur during training.
