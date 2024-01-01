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