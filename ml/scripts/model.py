import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib 

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = float('-inf')

    def train(self, X_train, X_test, y_train, y_test):
        """Train the model with multiple models and hyperparameters and return the best model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'random_forest': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost' : {
                'model': xgb.XGBRegressor(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                }
            }
        }

        results = {}

        for name, model_info in models.items():
            grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'model': best_model,
                'mse': mse,
                'r2': r2,
                'best_params': grid_search.best_params_
            }

            print(f'{name} Results:')
            print(f'MSE: {mse:.4f}')
            print(f'R2: {r2:.4f}')
            print(f'Best Params: {grid_search.best_params_}')
            
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = best_model

        return results
    
    def get_feature_importance(self, feature_names):
        """Get the feature importance of the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            return importance_df.sort_values(by='importance', ascending=False)
        else:
            print("The best model does not have feature importances.")
            return None
        
    def save_model(self, path="model"):
        """Save the best model to a file"""
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{path}/model.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
            print(f"Model and scaler saved to {path}")
        else:
            print("No model to save - BAD")

    def predict(self, X):
        """Predict the target variable for new data"""
        if self.best_model is None:
            raise ValueError("No model to make predictions.")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
def main():
    features_df = pd.read_csv("data/processed_features.csv")
    X = features_df.drop(columns=["cringe_score"], axis=1)
    y = features_df["cringe_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer()
    results = trainer.train(X_train, X_test, y_train, y_test)

    importance_df = trainer.get_feature_importance(X.columns)
    if importance_df is not None:
        print(importance_df)

    trainer.save_model()

    #Testign
    sample_post = X_test.iloc[0].to_frame().T
    predictions = trainer.predict(sample_post)
    print(predictions)

if __name__ == "__main__":
    main()


