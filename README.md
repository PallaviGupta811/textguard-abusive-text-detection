# TextGuard — Abusive Text Detection

A machine learning project that classifies text comments as abusive or non-abusive using character-level TF-IDF features and Logistic Regression. The project includes model comparison, held-out test evaluation, and an interactive Streamlit dashboard for experimenting with text classification.

## Overview

Online platforms receive large volumes of user-generated content, making potentially abusive language difficult to identify manually at scale. TextGuard explores how Natural Language Processing (NLP) and traditional machine learning can help flag potentially abusive comments.

The project uses labelled Wikipedia comments to train and evaluate text classification models. A web interface allows users to submit text, view the model's prediction score, and explore recent analyses during their session.

**This is an educational and academic project, not a production-ready content moderation system.**

## Features

- Text classification into abusive and non-abusive categories.
- Character-level TF-IDF feature extraction.
- Logistic Regression with class balancing.
- Comparison of word-level, character-level, and combined TF-IDF approaches.
- Validation-based threshold selection.
- Final evaluation on a held-out test set.
- Confusion matrix and model comparison visualizations.
- Interactive Streamlit dashboard with example comments.
- Session-based analysis history and CSV export.
- Model performance information and limitations displayed in the app.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, Logistic Regression
- **NLP / Feature Extraction:** TF-IDF
- **Data Processing:** pandas, NumPy, SciPy
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** joblib

## Dataset

The project uses the Jigsaw Toxic Comment Classification Challenge dataset, containing Wikipedia comments annotated across six categories:

- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

For this project, the six labels are combined into a binary target:

- **Abusive (1):** At least one of the six labels is positive.
- **Non-abusive (0):** All six labels are negative.

This is an operational definition based on the dataset's annotations. It does not represent every possible form of abusive language or establish that a comment is harmful in every context.

The dataset files are not included in this repository. Obtain the dataset from its original competition source and review the applicable terms before using it:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

After downloading and extracting the data, place the training CSV at:

`data/raw/train.csv`

## Methodology

### 1. Data preparation

- Inspected the dataset and checked for missing values and duplicate records.
- Combined the six toxicity labels into one binary target.
- Created stratified training, validation, and test splits using random seed `42`.
- Kept the test split separate from model and decision-threshold selection.

### 2. Model comparison

Three TF-IDF-based approaches were evaluated:

1. **Word-level TF-IDF:** Word unigrams and bigrams.
2. **Character-level TF-IDF:** Character n-grams using the `char_wb` analyzer.
3. **Combined TF-IDF:** Word-level and character-level features together.

Each approach uses Logistic Regression with class balancing. The validation set was used to compare models and select the decision threshold.

The selected model uses character-level TF-IDF with character n-grams of lengths 3–5, followed by Logistic Regression. Its selected decision threshold is `0.70`.

### 3. Final evaluation

The selected configuration was trained on the combined training and validation data and evaluated on the held-out test set.

## Results

### Final model — held-out test set

| Metric | Score |
|---|---:|
| Accuracy | 95.87% |
| Precision | 79.79% |
| Recall | 79.50% |
| F1-score | 79.65% |

The test set contained 23,936 comments, including 2,434 comments labelled abusive and 21,502 labelled non-abusive.

### Model comparison — validation set

| Model | Best tested F1-score |
|---|---:|
| Word-level TF-IDF | 77.26% |
| Character-level TF-IDF | 80.17% |
| Combined word + character TF-IDF | 80.03% |

The character-level model was selected because it achieved the highest F1-score among the tested validation configurations. These validation scores are separate from the final test results above.

### Visualizations

**Model comparison**

![Validation F1-score comparison](reports/figures/model_comparison.png)

**Final model confusion matrix**

![Confusion matrix for the held-out test set](reports/figures/confusion_matrix.png)

## Project Structure

```text
Abusive-Text-Detection/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/                  # Downloaded source dataset (not tracked)
│   └── processed/            # Prepared splits (not tracked)
├── models/
│   └── abusive_text_model.joblib
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       └── model_comparison.png
└── src/
    ├── inspect_dataset.py
    ├── prepare_dataset.py
    ├── train_baseline.py
    ├── train_char_model.py
    ├── train_combined_model.py
    ├── evaluate_final.py
    ├── save_model.py
    ├── plot_confusion_matrix.py
    └── plot_model_comparison.py
```

## Getting Started

### Prerequisites

- Python compatible with the pinned dependencies in `requirements.txt`
- pip
- The dataset downloaded and placed in the expected location if you intend to rerun training

### 1. Clone the repository

Replace the placeholder URL with your GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation for alternatives or use the Command Prompt activation script.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

From the repository root:

```powershell
streamlit run app.py
```

Streamlit will provide a local URL to open in your browser.

The saved model is included in the repository, so the dashboard can run without retraining, provided the dependencies are compatible.

## Reproducing the Experiments

After placing the source dataset at `data/raw/train.csv`, run the scripts from the repository root in this order:

```powershell
python src/inspect_dataset.py
python src/prepare_dataset.py
python src/train_baseline.py
python src/train_char_model.py
python src/train_combined_model.py
python src/evaluate_final.py
python src/save_model.py
python src/plot_confusion_matrix.py
python src/plot_model_comparison.py
```

The training and preparation scripts may take time depending on your hardware. Review each script's output and configuration if you want to reproduce or modify the experiments.

## Limitations and Responsible Use

- The training data consists of labelled English Wikipedia comments and may not represent other platforms, languages, communities, or communication styles.
- Dataset labels reflect the annotation scheme and may contain bias or ambiguity.
- The classifier can produce false positives and false negatives.
- Character patterns can help identify spelling variations but do not fully capture meaning, intent, sarcasm, or context.
- The displayed prediction score should not automatically be interpreted as a calibrated probability.
- The threshold of `0.70` represents a decision setting selected from the tested validation thresholds; it is not a universal boundary for abusive language.
- The model should not be used as the sole basis for account penalties, censorship, or other consequential decisions.

Human review and additional testing are necessary before considering deployment on a real platform.

## Future Improvements

- Evaluate on comments from additional domains and communities.
- Investigate multilingual and context-aware models.
- Perform error analysis and bias evaluation across different language patterns.
- Assess probability calibration and threshold trade-offs.
- Explore more comprehensive automated tests and deployment options.

## License

The project code does not yet specify a repository-level license. Before adding one, choose a license appropriate for your own contributions and review the applicable terms for the dataset, dependencies, and other included artifacts. Do not assume that all project components share the same license.

## Acknowledgements

- Jigsaw Toxic Comment Classification Challenge for the dataset.
- The open-source Python ecosystem, including scikit-learn, pandas, NumPy, SciPy, Matplotlib, joblib, and Streamlit.

---

**TextGuard is an academic machine learning project exploring abusive-text classification, model evaluation, and interactive NLP applications.**