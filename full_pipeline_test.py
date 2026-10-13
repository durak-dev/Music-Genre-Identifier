import pandas as pd
from transformers import pipeline


df = pd.read_csv('data/train.csv', index_col=0)
classification_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
