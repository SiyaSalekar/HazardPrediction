# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from scikit-learn
import seaborn as sns
import matplotlib.pyplot as plt

# Load sample dataset 
data = pd.DataFrame({
    'Temperature': [25.0, 30.0, 35.0, 40.0, 20.0, 15.0],
    'ChemicalSpills': [0, 1, 0, 1, 0, 0],
    'StructuralDamage': [0, 0, 1, 0, 0, 1],
    'UnauthorizedPersonnel': [0, 0, 0, 1, 0, 0],
    'Hazard': ['No Hazard', 'Fire', 'Structural Damage', 'Unauthorized Personnel', 'No Hazard', 'Structural Damage']
})

# Handling missing values by imputation or dropping rows/columns
data['Temperature'].fillna(data['Temperature'].mean(), inplace=True)

# Encode the 'Hazard' column to numerical labels
label_encoder = LabelEncoder()
data['Hazard'] = label_encoder.fit_transform(data['Hazard'])

# Split data into features and output variable
X = data[['Temperature', 'ChemicalSpills', 'StructuralDamage', 'UnauthorizedPersonnel']]
y = data['Hazard']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)