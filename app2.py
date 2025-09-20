import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume you have a dataframe called 'df' with your raw data
# Load your data (replace with your actual data loading)
# df = pd.read_csv('your_data.csv')

# --- Example Data Creation (replace with your actual data) ---
data = {'area': [1000, 1500, 2000, 2500, 3000],
        'bedrooms': [2, 3, 3, 4, 4],
        'bathrooms': [1, 2, 2, 3, 3],
        'mainroad': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'basement': ['No', 'Yes', 'No', 'No', 'Yes'],
        'parking': [1, 2, 2, 3, 3],
        'city': ['A', 'B', 'A', 'C', 'B'],
        'price': [100000, 150000, 200000, 250000, 300000]}
df = pd.DataFrame(data)

# --- Preprocessing (example: one-hot encoding) ---
# Separate features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Apply one-hot encoding to categorical features
X_encoded = pd.get_dummies(X, columns=['mainroad', 'basement', 'city'])

# Get the list of column names after encoding
model_columns = X_encoded.columns.tolist()

# --- Save the list of column names ---
MODEL_COLUMNS_PATH = "model_columns.joblib"
joblib.dump(model_columns, MODEL_COLUMNS_PATH)

print(f"'{MODEL_COLUMNS_PATH}' created successfully with the following columns:")
print(model_columns)

# --- Train your model (example: using Linear Regression) ---
# from sklearn.linear_model import LinearRegression
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#
# # Initialize and train the model
# # model = LinearRegression()
# # model.fit(X_train, y_train)
#
# # --- Save the trained model ---
# # MODEL_PATH = "model.joblib"
# # joblib.dump(model, MODEL_PATH)
# # print(f"'{MODEL_PATH}' created successfully.")
