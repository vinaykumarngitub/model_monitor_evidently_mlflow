# ML Model Drift Monitoring with Evidently and MLflow

This project contains two Jupyter notebooks demonstrating how to detect **Data Drift** and **Concept Drift** in a machine learning model using the `evidently` library and log the results using `mlflow`.

The experiments use the classic Iris dataset, a simple logistic regression model, and `ngrok` to expose the MLflow UI from a Colab environment.

## 🧪 Key Libraries

* **pandas**: For data manipulation.
* **scikit-learn**: For loading the Iris dataset, splitting data, and training a `LogisticRegression` model.
* **evidently**: For generating data drift and model performance reports.
* **mlflow**: For logging experiment parameters, metrics, and artifacts (the reports).
* **pyngrok**: To create a public URL for the MLflow UI when running in Google Colab.

---

## Notebook 1: `Evidently_mlflow_Lab_Data_Drift.ipynb`

This notebook demonstrates how to detect **Data Drift**. Data Drift occurs when the statistical properties of the input features (X) change between the training (reference) dataset and the production (current) dataset.

### Workflow

1.  **Setup**: Installs and imports all necessary libraries.
2.  **Load Data**: Loads the Iris dataset.
3.  **Split Data**: The dataset is split 50/50 into a `reference_df` (simulating training data) and a `current_df` (simulating production data).
4.  **Simulate Data Drift**: Data drift is intentionally introduced into the `current_df` by multiplying the 'sepal length (cm)' feature by 1.5.
5.  **Model Training**: A `LogisticRegression` model is trained *only* on the `reference_df`.
6.  **Generate Predictions**: The trained model is used to make predictions on both the reference and current datasets.
7.  **Generate Report**: An `evidently` report is created using `DataDriftPreset` and `ClassificationPreset` to compare the `current_df` to the `reference_df`.
8.  **Log to MLflow**:
    * An MLflow run is started.
    * Model parameters (features used) are logged.
    * Metrics (accuracy, precision, etc.) and drift information from the Evidently report are extracted and logged to MLflow.
    * The full HTML report (`drift_and_performance_report.html`) is saved and logged as an MLflow artifact.
9.  **Expose MLflow UI**: `ngrok` is used to create a public URL to view the MLflow dashboard.

---

## Notebook 2: `Evidently_mlflow_Lab_Concept_Drift.ipynb`

This notebook demonstrates how to detect **Concept Drift**. Concept Drift occurs when the relationship between the input features (X) and the target variable (y) changes over time.

### Workflow

1.  **Setup**: Installs and imports all necessary libraries.
2.  **Load Data**: Loads the Iris dataset.
3.  **Split Data**: The dataset is split 50/50 into a `reference_df` and `current_df`.
4.  **Simulate Concept Drift**: Concept drift is introduced by swapping the labels for class `1` and class `2` in the `current_df`. The feature distributions themselves are *not* changed.
5.  **Model Training**: A `LogisticRegression` model is trained *only* on the `reference_df` (which has the *original* feature-to-label mapping).
6.  **Generate Predictions**: The trained model is used to make predictions on both datasets. It is expected to perform poorly on the `current_df` because the underlying concept has changed.
7.  **Generate Report**: An `evidently` report is created to detect the drop in model performance and analyze the data.
8.  **Log to MLflow**:
    * An MLflow run is started.
    * Model parameters are logged.
    * Key metrics (accuracy, precision, etc.) from the Evidently report are extracted and logged to MLflow, highlighting the performance drop in the "current" data.
    * The full HTML report (`concept_drift_report.html`) is saved and logged as an MLflow artifact.
9.  **Expose MLflow UI**: `ngrok` is used to create a public URL to view the MLflow dashboard.

## 🚀 How to Run

1.  **Install Requirements**: The first cell in each notebook installs all required Python packages.
    ```python
    !pip install pandas scikit-learn mlflow evidently pyngrok
    ```
2.  **Run Cells**: Execute the cells in order.
3.  **Ngrok (Colab Only)**: If running in Google Colab, you will be prompted to paste an `ngrok` authtoken the first time you run it.
4.  **View Results**:
    * The Evidently HTML report will be saved to the local filesystem.
    * The MLflow UI will be accessible via the public `ngrok` URL printed at the end of the notebook. You can view all logged parameters, metrics, and the full HTML report in the "Artifacts" section of the MLflow run.
