from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent



# Step 1: Load and prepare your data
# (This is just an example, replace with your actual data loading code)
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y

# Step 2: Create and train your model
def train_model():
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (important for many ML algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy_pretuning = model.score(X_test_scaled, y_test)
    print(f"Model accuracy before tuning: {accuracy_pretuning:.4f}")
    
    # Optional: Hyperparameter tuning with GridSearchCV
    
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': range(1, 100, 10),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
        
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the best model
    best_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"Best model accuracy: {best_accuracy:.4f}")
    
    
    joblib.dump(best_model, APP_DIR / "model.joblib")
    joblib.dump(scaler, APP_DIR / "scaler.joblib")
    print("Model and scaler saved to disk.")

    return best_model, scaler


if __name__ == "__main__":
    train_model()