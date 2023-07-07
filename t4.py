import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
dataset = pd.read_csv(r"C:\Users\Abhishek\Downloads\Applications_for_Machine_Learning_internship_edited.xlsx - Sheet1.csv")

# Step 2: Select the desired columns
selected_columns = ['Python (out of 3)', 'Machine Learning (out of 3)', 'Natural Language Processing (NLP) (out of 3)',
                    'Deep Learning (out of 3)', 'Performance_UG', 'Performance_12', 'Performance_10']

# Step 3: Remove other columns from the dataset
dataset = dataset[selected_columns]

# Step 4: Split the dataset into training and testing sets
features = selected_columns[:-1]  # Exclude the target column
target = selected_columns[-1]  # Last column is the target

X = dataset[features]
y = dataset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a new model
model = RandomForestClassifier()

# Step 6: Train the model on the training dataset
model.fit(X_train, y_train)

# Step 7: Evaluate the performance of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

# Step 8: Select the intern with the highest score in specified features
selected_intern = dataset.loc[dataset[features].sum(axis=1).idxmax()]

# Step 9: Output the selected intern's name and number in the dataset
print('Best Intern Name:', selected_intern[target])
print('Number in Dataset:', selected_intern.name)
