import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the dataset from the URL and save it in the 'data/' folder
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# You can alternatively save the file locally like this:
# df.to_csv('data/BostonHousing.csv', index=False)

# Define features and target
X = df.drop('medv', axis=1)  # 'medv' is the target variable (median value of homes)
y = df['medv']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up an MLflow experiment
mlflow.set_experiment("Housing_Price_Prediction")

def train_model(model_name, model):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric('mse', mse)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} MSE: {mse}")
        
        return mse, run.info.run_id  

# Train Linear Regression model
lr = LinearRegression()
mse_lr, run_id_lr = train_model("Linear_Regression", lr)

# Train Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf, run_id_rf = train_model("Random_Forest", rf)

# Log the best model (Random Forest) to the Model Registry
if mse_rf < mse_lr:
    print(f"Registering Random Forest model with MSE: {mse_rf}")
    mlflow.register_model(
        f"runs:/{run_id_rf}/model",  # Using the run_id of the best model (Random Forest)
        "Best_Housing_Model"  # Model name in the registry
    )
else:
    print(f"Registering Linear Regression model with MSE: {mse_lr}")
    mlflow.register_model(
        f"runs:/{run_id_lr}/model",
        "Best_Housing_Model"
    )
