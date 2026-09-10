#Pandas 10/09/2026
#Miami Housing Price Analysis: Explore housing data, identify price trends, and build simple price prediction model using decision trees.
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

miami_data_path = Path(__file__).with_name("miami-housing.csv")
df = pd.read_csv(miami_data_path)
df = df.dropna()  # Drop rows with missing values
#print(df.head(10))  # Display the first 10 rows of the DataFrame
miami_data = df
miami_data.describe()

y = miami_data.SALE_PRC
#Choosing Features 
miami_features = ['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'age', 'LATITUDE', 'LONGITUDE']
#Selecting x to model the features
x = miami_data[miami_features]

#Define model
model = DecisionTreeRegressor(random_state=1)

#Fit model 
model.fit(x, y)

#Testing the model rq
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(model.predict(x.head()))

#Model Validation using Mean Absolute Error (MAE)
#To calculate the MAE
from sklearn.metrics import mean_absolute_error
predictions = model.predict(x)
mae = mean_absolute_error(y, predictions)
print(f"Mean Absolute Error: {mae}")

#Import train_test_split to split the data into training and testing sets
x_train, x_valid, y_train, y_valid = train_test_split(
	x, y, test_size=0.25, random_state=1
)
#Define model 
model = DecisionTreeRegressor(random_state=1)
#Model Fit 
model.fit(x_train, y_train)

#Get predicted results from the validation data 
val_predictions = model.predict(x_valid)
print("Validation Predictions:")
print(mean_absolute_error(y_valid, val_predictions))