import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\enosh\Desktop\VINAY SIR ASSIGNMENT\mlops_pipeline\data\data.csv')

X = df[["area", "bedrooms",]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)     

with open(r'C:\Users\enosh\Desktop\VINAY SIR ASSIGNMENT\mlops_pipeline\backend\model.pkl', 'wb') as f:
    pickle.dump(model, f)