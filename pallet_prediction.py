import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming df is the DataFrame containing the data

# Extracting time-related features
df['Hour_of_Day'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month
df.drop(['Timestamp'], axis=1, inplace=True)  # Drop original Timestamp column

# Define numeric and categorical features
numeric_features = ['Distance_meters', 'Items_on_Pallet']
categorical_features = ['Warehouse_Zone', 'Location', 'Hour_of_Day', 'Day_of_Week', 'Month']  

# Create a ColumnTransformer for more complex preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Scaling numeric features
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
    ],
    remainder='drop'  # Drop any other columns not specified
)

# Append the model to the preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply preprocessing to the data
X_processed = pipeline.fit_transform(df)

# Split dataset into features and target variable
y = df['Retrieval_Time_minutes'].values  # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Model Initialization
models = {
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'Neural Network': MLPRegressor(random_state=42)
}

# Hyperparameter Grid for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Model Training and Evaluation
results = {}
for name, model in models.items():
    if name == 'Random Forest':
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(preprocessor.fit_transform(X_train), y_train)
        best_params = grid_search.best_params_
        model = RandomForestRegressor(**best_params, random_state=42)
    
    model.fit(preprocessor.fit_transform(X_train), y_train)
    predictions = model.predict(preprocessor.transform(X_test))
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    results[name] = {'Mean Squared Error (MSE)': mse, 'Root Mean Squared Error (RMSE)': rmse}

# Display evaluation metrics
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

#Evaluation with Additional Metrics
results = {}
for name, model in models.items():
    if name == 'Random Forest':
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(preprocessor.fit_transform(X_train), y_train)
        best_params = grid_search.best_params_
        model = RandomForestRegressor(**best_params, random_state=42)
    
    model.fit(preprocessor.fit_transform(X_train), y_train)
    predictions = model.predict(preprocessor.transform(X_test))
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {'Mean Squared Error (MSE)': mse, 'Root Mean Squared Error (RMSE)': rmse,
                     'Mean Absolute Error (MAE)': mae, 'R-squared': r2}
