import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pysr import PySRRegressor
from datetime import datetime

# Load and preprocess data
sys.path.append(os.path.abspath("external/AlphaPEM"))
data = pd.read_pickle(r"data/processed/validated_final_57344.pkl")
data = data[data['Ucell'].apply(lambda x: all(val > 0 for val in x))]


exploaded_df = data.explode(['ifc', 'Ucell'])

# Use current date-time as a run ID
now = datetime.now()
run_id = now.strftime("%Y%m%d_%H%M%S")

np.random.seed(42)

# Define input parameters
parameters = ['Tfc', 'i0_c_ref', 'Pa_des', 'kappa_c', 'kappa_co', 'tau', 'Re', 'ifc']

# Get unique curve IDs and split train/test
unique_ids = data['SHA256'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

# Filter rows based on train/test IDs
train_df = exploaded_df[exploaded_df['SHA256'].isin(train_ids)]
test_df = exploaded_df[exploaded_df['SHA256'].isin(test_ids)]

# Prepare features and targets
X_train = train_df[parameters].values
y_train = train_df['Ucell'].values
X_test = test_df[parameters].values
y_test = test_df['Ucell'].values

# Train symbolic regressor
model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "square", "cube", "inv", "neg"],
    elementwise_loss="L2DistLoss()",
    model_selection="accuracy",
    maxsize=20,
    maxdepth=5,
    verbosity=1,
    ncycles_per_iteration=30,
    output_directory=os.path.join("models", "pysr_equations"),
    batching=True,
    random_state=42,
    deterministic=True,
    parallelism="serial",
    run_id=run_id,
    batch_size=100
)

model.fit(X_train, y_train, variable_names=parameters)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")