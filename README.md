# ML Portfolio Demo

This repository contains two demonstration projects for an AI/ML portfolio.

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

## Hardware Note

These demonstrations were created on a machine with an RTX 3080 Ti GPU, Ryzen 9 CPU and 64 GB RAM. GPU acceleration is not strictly required for the logistic regression example but will be beneficial for larger deep‑learning models.

## License and Data Source

The sample dataset used here is a small synthetic subset meant for demonstration purposes. When using real datasets, please follow their respective licenses and terms of use. For example, the IMDb movie reviews dataset is available under the Apache 2.0 license via various public sources.
