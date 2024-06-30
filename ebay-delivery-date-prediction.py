# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset (replace 'dataset.csv' with your actual dataset file)
try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Dataset file 'dataset.csv' not found. Please provide the correct file path.")
    exit(1)  # Exit script if dataset is not found

# Preprocess data
# Assume 'seller_packaging_time', 'transit_time', and 'delivery_date' are columns in your dataset
X = data[['seller_packaging_time', 'transit_time']]  # Features
y = data['delivery_date']  # Target variable

# Split data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except NameError:
    print("Ensure you have imported train_test_split correctly from sklearn.model_selection.")
    exit(1)  # Exit script if train_test_split fails

# Initialize RandomForestRegressor
try:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    # Example prediction for new data
    new_data = pd.DataFrame({'seller_packaging_time': [2.5], 'transit_time': [1.8]})
    predicted_delivery_date = model.predict(new_data)
    print(f"Predicted Delivery Date: {predicted_delivery_date[0]} days")
except (ValueError, TypeError):
    print("Ensure that your dataset has appropriate columns and the RandomForestRegressor parameters are correctly set.")
