# =============================================================================
# CUSTOMER CHURN PREDICTION SCRIPT
# =============================================================================

# --- 1. Import Necessary Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Load and Preprocess Data ---
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Error: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    print("Please download the dataset and place it in the same directory as the script.")
    exit()

print("Initial data loaded successfully.")

# Convert 'TotalCharges' to a numeric type. Blank spaces will become NaN.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows that have any missing values (e.g., the newly created NaNs in TotalCharges)
df.dropna(inplace=True)

# Save the customerID column for the final output, then drop it for training.
# This ensures the IDs are perfectly aligned with the data used for prediction.
customer_ids = df['customerID']
df = df.drop('customerID', axis=1)

# --- 3. Feature Engineering ---

# Convert the target variable 'Churn' to a binary format (0 for 'No', 1 for 'Yes')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select all categorical columns (object type) for one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns

# Use one-hot encoding to convert categorical variables into a numerical format
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Data preprocessing and feature engineering complete.")
print(f"Shape of the final processed data: {df_encoded.shape}")

# Separate the features (X) from the target variable (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# --- 4. Model Training ---

# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete.")

# --- 5. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# --- 6. Visualization: Feature Importance ---

# Get feature importances from the trained model's coefficients
importances = model.coef_[0]
feature_names = X_train.columns

# Create a DataFrame for easy sorting and plotting
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top 10 most influential features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Features Influencing Churn')
plt.xlabel('Importance (Coefficient)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")
# To display the plot when running the script, uncomment the line below
# plt.show()


# --- 7. Final Output Generation ---

print("\n--- Generating Final Actionable Output File ---")

# Use the trained model to predict churn probabilities for the ENTIRE dataset
all_churn_probabilities = model.predict_proba(X)[:, 1]

# Create the final output DataFrame with customer IDs and their churn scores
output_df = pd.DataFrame({
    'customerID': customer_ids.values,
    'Churn_Probability': all_churn_probabilities
})

# Sort the results to show the highest-risk customers first
output_df = output_df.sort_values(by='Churn_Probability', ascending=False)

# Save the final list to a CSV file
output_df.to_csv('churn_predictions_output.csv', index=False)

print("Successfully created 'churn_predictions_output.csv'.")
print("This file contains all customers ranked by their churn risk.")

# Display the top 5 highest-risk customers from the final file
print("\nTop 5 Customers at Highest Risk:")
print(output_df.head())