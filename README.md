# Loan Default Prediction using MLP

## Project Overview

This project uses a Multi-Layer Perceptron (MLP), a neural network-based classifier, to predict the default risk of loan applicants using their financial and personal information.

**Prediction classes**
- `0` — Low Default Risk
- `1` — High Default Risk

## Objectives

- Understand and explore the loan dataset.
- Perform exploratory data analysis (EDA).
- Inspect missing values and duplicate records.
- Encode categorical variables and scale numerical features.
- Train an MLP classifier.
- Evaluate performance using classification metrics.
- Experiment with activation functions, hidden-layer structures, and learning rates.
- Predict the default-risk class for a new applicant.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- `MLPClassifier`

## Project Structure

```text
Loan-Default-Prediction-MLP/
├── assets/
├── data/
│   └── Loan_Default.csv
├── results/
├── src/
│   └── loan_default_prediction.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset Description

| Feature | Description |
|---|---|
| `Age` | Applicant's age |
| `Income` | Annual income |
| `LoanAmount` | Requested loan amount |
| `CreditScore` | Applicant's credit score |
| `EmploymentYears` | Years of employment |
| `ExistingLoans` | Number of existing loans |
| `MonthlyDebt` | Existing monthly debt |
| `LoanTerm` | Loan duration |
| `PreviousDefault` | Previous default history |
| `HomeOwnership` | Home ownership status, such as Rent, Own, or Mortgage |
| `Default` | Target variable: 0 or 1 |

## Model Configuration

The baseline model uses the following `MLPClassifier` configuration:

```python
MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)
```

## Project Workflow

1. Load and understand the dataset.
2. Perform exploratory data analysis.
3. Check missing values, duplicates, and target-class balance.
4. Separate input features (`X`) and target (`y`).
5. Split the data using stratified train-test sampling.
6. Impute missing values, encode categorical features, and scale numerical features.
7. Train the MLP classifier.
8. Evaluate accuracy, precision, recall, and F1-score.
9. Generate a confusion matrix, classification report, and training-loss plot.
10. Run hyperparameter experiments.
11. Predict the risk class for a new applicant.

## Hyperparameter Experiments

### Experiment 1: Activation Functions

- `identity`
- `logistic`
- `tanh`
- `relu`

### Experiment 2: Hidden-Layer Structures

- `(10,)`
- `(20, 10)`
- `(50, 25)`
- `(100, 50, 25)`

### Experiment 3: Learning Rates

- `0.0001`
- `0.001`
- `0.01`
- `0.1`

The program saves experiment metrics and plots to help compare configurations.

## Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted defaults that are actual defaults.
- **Recall:** Proportion of actual defaults correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of correct and incorrect predictions by class.
- **Training loss:** Loss recorded during model training.

Recall for class `1` is important because missed defaults may represent financial risk. Model performance should not be judged by accuracy alone.

## Installation and Execution

### 1. Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/Loan-Default-Prediction-MLP.git
cd Loan-Default-Prediction-MLP
```

### 2. Create and Activate a Virtual Environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file should include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### 4. Add the Dataset

Place the CSV file at:

```text
data/Loan_Default.csv
```

The filename must match the path configured in `src/loan_default_prediction.py`.

### 5. Run the Program

Run from the project root:

```bash
python src/loan_default_prediction.py
```

## Results

When the program runs successfully, it generates plots and reports in the `results/` directory, including:

- Default-class distribution
- Numerical feature distributions
- Correlation heatmap
- Missing-value visualization
- Confusion matrix
- Training-loss curve
- Activation-function comparison
- Hidden-layer comparison
- Learning-rate comparison
- Classification report and CSV metric summaries

Actual metrics depend on the dataset and will be generated during execution.

## Learning Outcomes

- Understanding neural networks and MLP classifiers.
- Performing EDA on structured financial data.
- Applying imputation, categorical encoding, and feature scaling.
- Understanding stratified train-test splitting.
- Evaluating binary classification models.
- Comparing neural-network hyperparameters.
- Visualizing model performance.

## Limitations

- Performance depends on the quality and representativeness of the dataset.
- Accuracy alone is not sufficient to evaluate loan-default prediction.
- Repeatedly selecting configurations based on test-set results can bias the reported evaluation; a validation set or cross-validation is preferable.
- This project is educational and is not validated for real-world lending decisions.

## Future Improvements

- Use cross-validation and systematic hyperparameter tuning.
- Evaluate ROC-AUC and precision-recall curves.
- Investigate class-imbalance handling methods.
- Add a Streamlit interface for interactive predictions.
- Explore model interpretability and fairness.

## Author

**Shreya Borate**  
Computer Science Graduate | Data Science | Machine Learning | Deep Learning

## License

An MIT License may be added by including a `LICENSE` file in the repository.
