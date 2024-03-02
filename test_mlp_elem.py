import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


df = pd.read_csv("complete.csv")
#df = df[['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B08', 'B09', 'B11', 'B12', 'oc']]
df = df[['EVI', 'oc']]

for column in df.columns:
    scaler = MinMaxScaler()
    scaled_column = scaler.fit_transform(df[[column]])
    df[column] = scaled_column.flatten()

data = df.to_numpy()
train_data, test_data = model_selection.train_test_split(data, test_size=0.1, random_state=2)
train_x = train_data[:,0:-1]
train_y = train_data[:,-1]
test_x = test_data[:,0:-1]
test_y = test_data[:,-1]
mlp_regressor = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_regressor = mlp_regressor.fit(train_x, train_y)
r2 = mlp_regressor.score(test_x, test_y)
print(r2)
