import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df['churn'] = df['churn'].map({'Yes':1,'No':0})
    X = df.drop(['churn','customer_id'],axis=1)
    y = df['churn']

    return train_test_split(X,y,test_size=0.2,random_state=42)
