# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic = pd.read_csv("titanic.csv")

# Quick overview of the dataset
print(titanic.info())
print(titanic.describe())

# Handle missing values
# Fill 'Age' with median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Fill 'Embarked' with the mode
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
titanic.drop(columns=['Cabin'], inplace=True)

# Drop rows with missing values in 'Fare' (if any)
titanic.dropna(subset=['Fare'], inplace=True)

# Data visualization
# Survival by gender
sns.countplot(data=titanic, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.savefig('survival_by_gender.png')
plt.show()

# Age distribution
sns.histplot(data=titanic, x='Age', hue='Survived', kde=True)
plt.title('Age Distribution by Survival')
plt.savefig('age_distribution.png')
plt.show()

# Class-wise survival rate
sns.barplot(data=titanic, x='Pclass', y='Survived')
plt.title('Survival Rate by Class')
plt.savefig('survival_by_class.png')
plt.show()

# Save cleaned data
titanic.to_csv("titanic_cleaned.csv", index=False)
print("Data cleaning and EDA completed. Cleaned dataset saved as 'titanic_cleaned.csv'.")