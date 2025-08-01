# ML Portfolio Demo

This repository contains several demonstration projects for an AI/ML portfolio. The goal is to showcase a range of machine‑learning techniques—from classical algorithms and data preprocessing to API deployment and deep‑learning workflows. Each demo is self‑contained and documented.

## 1. Sentiment Analysis Notebook

The Jupyter notebook `sentiment_analysis.ipynb` shows how to build a simple sentiment analysis pipeline:

- Load a dataset of movie reviews and sentiment labels (`imdb_reviews.csv`). For illustration, a small sample dataset is included. For real experiments, replace it with the full IMDb dataset or another dataset of your choice.
- Vectorize the text using TF‑IDF.
- Train a logistic regression classifier using scikit‑learn.
- Evaluate the model and display a confusion matrix.

### Requirements

Install required packages in a Python environment (e.g. conda):

```bash
conda create -n ml_demo python=3.11
conda activate ml_demo
pip install jupyter pandas scikit-learn matplotlib seaborn
```

Run the notebook with Jupyter and follow the instructions in the cells.

## 2. Prompt Engineering Example

The script `prompt_example.py` demonstrates how to use the OpenAI API to generate product descriptions using a language model. You need to set your `OPENAI_API_KEY` environment variable before running the script.

```bash
export OPENAI_API_KEY=your_api_key_here
python prompt_example.py
```

You can experiment with different prompts, temperatures and maximum token lengths to see how the outputs vary.

## 3. Iris Classification with Random Forest

The notebook `iris_random_forest.ipynb` demonstrates how to perform classical machine‑learning classification using the well‑known Iris flower dataset. The workflow includes:

- Loading the dataset via scikit‑learn and converting it into a pandas DataFrame.
- Exploratory data analysis and visualization (e.g. pairwise scatter plots of the features).
- Splitting the data into training and test sets.
- Training a RandomForestClassifier.
- Evaluating the model using accuracy and a confusion matrix.

This demo illustrates good practices for data exploration, model training and evaluation.

## 4. FastAPI Inference API

The script `fastapi_iris_api.py` shows how to build a simple REST API using FastAPI to serve an ML model. At startup it trains a logistic regression model on the Iris dataset and exposes an endpoint `/predict` that accepts feature values and returns the predicted class along with the probability. To run the API:

```bash
pip install fastapi uvicorn scikit-learn pydantic
uvicorn fastapi_iris_api:app --reload
```

Then send a POST request to `/predict` with JSON payload containing sepal and petal measurements. The API will return a JSON object with the predicted species and the associated probability.

## 5. Transfer Learning with PyTorch

The notebook `pytorch_transfer_learning.ipynb` illustrates how to leverage a pre‑trained ResNet‑18 model for image classification on the CIFAR‑10 dataset. Steps include:

- Loading and transforming the CIFAR‑10 dataset.
- Initializing a pre‑trained ResNet‑18 from torchvision and freezing its convolutional layers.
- Replacing the final fully connected layer to match the number of classes.
- Training the modified network for a few epochs.
- Evaluating performance on the test set.

This demo requires a GPU (e.g. RTX 3080 Ti) for efficient training and uses PyTorch and torchvision.

## Hardware Note

These demonstrations were created on a machine with an RTX 3080 Ti GPU, Ryzen 9 CPU and 64 GB RAM. GPU acceleration is not strictly required for the logistic regression example but will be beneficial for larger deep‑learning models.

## License and Data Source

The sample dataset used here is a small synthetic subset meant for demonstration purposes. When using real datasets, please follow their respective licenses and terms of use. For example, the IMDb movie reviews dataset is available under the Apache 2.0 license via various public sources.
