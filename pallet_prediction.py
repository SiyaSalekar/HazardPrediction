import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assume df is your DataFrame with the necessary features and target variable
# ...
import pandas as pd
import numpy as np

# Generating sample data for pallet retrieval
np.random.seed(42)
num_samples = 1000
data = {
    'Warehouse_Zone': np.random.choice(['A', 'B', 'C', 'D'], size=num_samples),
    'Timestamp': pd.date_range('2022-01-01', periods=num_samples, freq='H'),
    'Distance_meters': np.random.randint(20, 100, size=num_samples),
    'Items_on_Pallet': np.random.randint(5, 25, size=num_samples),
    'Retrieval_Time_minutes': np.random.randint(10, 30, size=num_samples)
}

# Create aisle, row, and column based on Warehouse_Zone
def generate_location(zone):
    if zone == 'A':
        aisle, row, column = '1', '1', np.random.choice(['1', '2', '3'])
    elif zone == 'B':
        aisle, row, column = '2', '2', np.random.choice(['1', '2', '3'])
    elif zone == 'C':
        aisle, row, column = '3', '3', np.random.choice(['1', '2', '3'])
    else:  # Zone D
        aisle, row, column = '4', '4', np.random.choice(['1', '2', '3'])
    return f"Aisle {aisle}, Row {row}, Column {column}"

data['Location'] = [generate_location(zone) for zone in data['Warehouse_Zone']]

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Display the sample data
print(df.head())


# Preprocessing
df['Hour_of_Day'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month
df.drop(['Timestamp'], axis=1, inplace=True)

for col in ['Warehouse_Zone', 'Location', 'Hour_of_Day', 'Day_of_Week', 'Month']:
    df[col] = df[col].astype('category')

numeric_features = ['Distance_meters', 'Items_on_Pallet']
categorical_features = ['Warehouse_Zone', 'Location', 'Hour_of_Day', 'Day_of_Week', 'Month']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='drop'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(df)
X_processed_df = pd.DataFrame.sparse.from_spmatrix(X_processed)

# Train-test split
y = df['Retrieval_Time_minutes'].values
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
predictions = rf_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# Plotting predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Retrieval Time (minutes)')
plt.ylabel('Predicted Retrieval Time (minutes)')
plt.title('Random Forest: Actual vs Predicted')
plt.show()

import matplotlib.pyplot as plt


zone_counts = df['Warehouse_Zone'].value_counts()
plt.pie(zone_counts, labels=zone_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Samples across Warehouse Zones')
plt.show()

# Display evaluation metrics
# for name, metrics in results.items():
#     print(f"{name}:")
#     for metric, value in metrics.items():
#         print(f"  {metric}: {value}")

# #Evaluation with Additional Metrics
# results = {}
# for name, model in models.items():
#     if name == 'Random Forest':
#         grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
#         grid_search.fit(preprocessor.fit_transform(X_train), y_train)
#         best_params = grid_search.best_params_
#         model = RandomForestRegressor(**best_params, random_state=42)
    
#     model.fit(preprocessor.fit_transform(X_train), y_train)
#     predictions = model.predict(preprocessor.transform(X_test))
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     results[name] = {'Mean Squared Error (MSE)': mse, 'Root Mean Squared Error (RMSE)': rmse,
#                      'Mean Absolute Error (MAE)': mae, 'R-squared': r2}
