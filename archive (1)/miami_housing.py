#Pandas 10/09/2026
#Miami Housing Price Analysis: Explore housing data, identify price trends, and build simple price prediction model using decision trees.
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = Path(__file__).with_name("miami-housing.csv")
df = pd.read_csv(csv_path)
#print(df.isnull().sum())  # Check for missing values in the DataFrame
df = df.dropna()  # Drop rows with missing values
print(df.head(10))  # Display the first few rows of the DataFrame


