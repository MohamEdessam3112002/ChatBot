import pandas as pd
from langchain_community.document_loaders import CSVLoader

df = pd.read_csv("C:/Users/DELL/Desktop/Python crash course/Langchain/products.csv")
print(df.head())

loader = CSVLoader("products.csv")
documents = loader.load()

print(documents[0])