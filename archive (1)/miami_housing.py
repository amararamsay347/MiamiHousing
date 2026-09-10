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
x = miami_data[miami_features]
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size=0.2, random_state=1
)
model = DecisionTreeRegressor(random_state=1)

#Fit model 
model.fit(x_train, y_train)

#Testing the model rq
print("Making predictions for the following 5 houses:")
print(x_test.head())
print("The predictions are")
print(model.predict(x_test.head()))


