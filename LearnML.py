import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
#print(data.describe()) ## describe() is similar to summary function in R (gives a summary of the df)
#print(data.columns) ## Prints all the column names
#print(data.shape) ##Prints
data_price = data.SalePrice #Assign individual columns to a variable
#print(data_price.head(10))  #similar to head funtion in R

columns = ['BedroomAbvGr','FullBath','YearBuilt',
               'OverallQual','OverallCond','LotArea'] #defining  columns to be selected

select_columns = data[columns] #Selecting columns from the test df
#printing the summary of the selected columns

##########################################################################################
########################### First lesson is using Python for Machine Learning ############

y = data.SalePrice #This is the target variable in the modeling

test_predictors = ['BedroomAbvGr','1stFlrSF','2ndFlrSF',
                   'FullBath','YearBuilt','OverallQual','OverallCond','LotArea']

### I am choosing to ignore the fact that there may be some co-linearity among some of the
### predictors I have chose

X = data[test_predictors] #These are variable affecting pricing


# Define model
test_model = DecisionTreeRegressor()

# Fit model
test_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predicted prices are")
print(test_model.predict(X.head()))

#Find the error in the model

# This uses the home prices from the predicted model and compares them with
# the home prices in the training dataset
predicted_home_prices = test_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
test_model2 = DecisionTreeRegressor()
# Fit model
test_model2.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = test_model2.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

##Trying to figure out the number of iterations required to achieve the perfect model
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
################################################################################################
## Let's build a model using Random Forest Regression.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
randomF_preds = forest_model.predict(val_X)

print("The Mean Absolute Error for the Random Forest model is:")
print(mean_absolute_error(val_y, randomF_preds))
