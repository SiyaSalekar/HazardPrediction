import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate sample random data
np.random.seed(42)
num_samples = 1000

data = {
    'Weight': np.random.choice(['heavy', 'light'], size=num_samples),
    'Aisle': np.random.choice(['1', '2', '3', '4'], size=num_samples),
    'Row': np.random.choice(['1', '2', '3'], size=num_samples),
    'Column': np.random.choice(['1', '2', '3'], size=num_samples),
    'Distance_from_Entry': np.random.randint(10, 100, size=num_samples),
    'Time_of_Day': np.random.choice(['morning', 'afternoon', 'night'], size=num_samples),
    'Day_of_Week': np.random.choice(['weekday', 'weekend'], size=num_samples),
    'Congestion': np.random.choice(['low', 'medium', 'high'], size=num_samples),
    'Previous_Retrieval_Time': np.random.randint(10, 30, size=num_samples),
    'Temperature': np.random.choice(['normal', 'cold', 'hot'], size=num_samples),
    'Automation_Level': np.random.choice(['low', 'medium', 'high'], size=num_samples),
}

df = pd.DataFrame(data)

# Convert categorical columns to 'category' type 
categorical_cols = ['Weight', 'Aisle', 'Row', 'Column', 'Time_of_Day', 'Day_of_Week', 'Congestion', 'Temperature', 'Automation_Level']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop('Previous_Retrieval_Time', axis=1)
y = df_encoded['Previous_Retrieval_Time']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest model based on research show the highest accuracy with dynamic warehouse data
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# predictions on the test set
predictions = rf_model.predict(X_test)


