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
print(miami_data.describe())

y = miami_data.SALE_PRC
#Choosing Features 
miami_features = ['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'age', 'LATITUDE', 'LONGITUDE']
#Selecting x to model the features
x = miami_data[miami_features]

#Model Validation using Mean Absolute Error (MAE)
#To calculate the MAE
from sklearn.metrics import mean_absolute_error

#Import train_test_split to split the data into training and testing sets
x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.25, random_state=1
)

#Define model, this helps with overfitting and underfitting of the model.
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_leaf=10,
    random_state=1
)
#Model Fit 
model.fit(x_train, y_train)

#Get predicted results from the validation data 
val_predictions = model.predict(x_valid)
print("Making predictions for the following 5 houses:")
print(x_valid.head())
print("The predictions are")
print(val_predictions[:5])

comparison = pd.DataFrame({
    "Actual Price": y_valid.head().values,
    "Predicted Price": val_predictions[:5]
})
print("Actual versus predicted prices for Decision Tree Model:")
print(comparison)

print("Mean Absolute Error of Validation Predictions for Decision Tree Model:")
print(mean_absolute_error(y_valid, val_predictions))

#Use validation data to curb overfitting and improve model performance using Decision Trees.
def get_mae(max_leaf_nodes, x_train, x_valid, y_train, y_valid):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(x_train, y_train)
    preds_val = model.predict(x_valid)
    mae = mean_absolute_error(y_valid, preds_val)
    return mae

#Compare MAE with different values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, x_train, x_valid, y_train, y_valid)
    print(f"Max leaf nodes: {max_leaf_nodes} \t\t Mean Absolute Error: {my_mae}") 
#500 is the best value for max_leaf_nodes, as it has the lowest MAE.

#To improve, use the Random Forest Regressor to improve the model performance and reduce overfitting.
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=1,
    n_jobs=-1
) #AI suggested increasing the no of estimators and max depth to reduce overfitting. 
#TO reduce overfitting, increase the number of estimators and set a maximum depth for the trees.
forest_model.fit(x_train, y_train)
model_predictions = forest_model.predict(x_valid)
#print("Mean Absolute Error of Random Forest Model Predictions:", mean_absolute_error(y_valid, model_predictions))
#Random Forest Model has a lower MAE than the Decision Tree Model,better performance and reduced overfitting.

forest_train_predictions = forest_model.predict(x_train)

print(
    "Random Forest Training MAE:",
    mean_absolute_error(y_train, forest_train_predictions)
)
print(
    "Random Forest Validation MAE:",
    mean_absolute_error(y_valid, model_predictions)
)

print("Actual versus predicted prices for Random Forest Model:")
comparison_rfm = pd.DataFrame({
    "Actual Price": y_valid.head().values,
    "Predicted Price": model_predictions[:5]
})
print(comparison_rfm)
