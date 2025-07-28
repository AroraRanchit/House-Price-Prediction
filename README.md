# House-Price-Prediction
This is a project created with the purpose of learning to implement various ML methods and test them.


import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score


class HousePricePredictor:
    def __init__(self, model_path='house_price_model_xgboost.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.X = None
        self.y = None

    # === Step 1: Extract Data ===
    def extract_data(self, file_path, sheet='Sheet1'):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            print(f"✅ Data loaded from '{file_path}' with shape {df.shape}")
            return df
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"❌ Error while reading Excel file: {e}")

    # === Step 2: Clean & Remove Outliers ===
    def exploratory_data_processing(self, df):
        try:
            df = df.drop(columns=['Id'], errors='ignore')
            df = df.dropna(subset=['SalePrice'])

            Q1 = df['SalePrice'].quantile(0.25)
            Q3 = df['SalePrice'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df['SalePrice'] >= lower) & (df['SalePrice'] <= upper)]

            print(f"🧹 Data cleaned and outliers removed. Remaining rows: {df.shape[0]}")
            return df
        except KeyError:
            print("❌ 'SalePrice' column not found in the dataset.")
        except Exception as e:
            print(f"❌ Error during exploratory data processing: {e}")

    # === Step 3: Preprocess Features ===
    def preprocess_features(self, df):
        try:
            self.X = df.drop(columns=['SalePrice'])
            self.y = df['SalePrice']

            numeric_cols = self.X.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = SimpleImputer(strategy='mean')
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

            print("⚙️ Feature preprocessing setup complete.")
            return preprocessor
        except KeyError:
            print("❌ 'SalePrice' column not found.")
        except Exception as e:
            print(f"❌ Error in preprocessing features: {e}")

    # === Step 4: Calculate MAPE ===
    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # === Step 5: Train and Evaluate Model ===
    def train_and_evaluate(self, preprocessor):
        try:
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1,
                                           max_depth=4, random_state=42))
            ])

            self.pipeline.fit(self.X, self.y)
            joblib.dump(self.pipeline, self.model_path)

            predictions = self.pipeline.predict(self.X)
            mse = mean_squared_error(self.y, predictions)
            r2 = r2_score(self.y, predictions)
            mape = self.calculate_mape(self.y, predictions)

            self.log_results(predictions, self.y, mse, r2, mape)
            self.save_predictions(predictions)

        except Exception as e:
            print(f"❌ Error during model training/evaluation: {e}")

    # === Logging & Result Display ===
    def log_results(self, predictions, y_true, mse, r2, mape):
        print("\n=== Predicted vs Actual Prices ===")
        for i, (pred, actual) in enumerate(zip(predictions[:9], y_true[:9])):
            print(f"Row {i+1}: Predicted: ${pred:,.2f} | Actual: ${actual:,.2f}")

        print(f"\n📊 Mean Squared Error (MSE): {mse:,.2f}")
        print(f"📏 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"✅ R² Score (Accuracy): {r2 * 100:.2f}%")
        print(f"💾 Model saved to '{self.model_path}'")

    # === Save Prediction Output ===
    def save_predictions(self, predictions, filename='predicted_vs_actual_xgboost.csv'):
        try:
            df_result = self.X.copy()
            df_result['ActualPrice'] = self.y.values
            df_result['PredictedPrice'] = predictions
            df_result.to_csv(filename, index=False)
            print(f"📁 Predictions saved to '{filename}'")
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")

    # === Predict New Data ===
    def predict_new_data(self, new_data_path):
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Trained model not found at '{self.model_path}'.")
                return

            model = joblib.load(self.model_path)
            new_df = pd.read_excel(new_data_path)
            predictions = model.predict(new_df)

            print("\n🔮 Predicted Prices for New Data:")
            for i, price in enumerate(predictions):
                print(f"Row {i+1}: ${price:,.2f}")

            return predictions
        except FileNotFoundError:
            print(f"❌ File '{new_data_path}' not found.")
        except Exception as e:
            print(f"❌ Error during prediction: {e}")


# === Main Execution ===
if __name__ == "__main__":
    predictor = HousePricePredictor()

    file_path = 'HousePricePrediction.xlsx'

    df = predictor.extract_data(file_path)
    if df is not None:
        df = df.iloc[:1461]
        df_cleaned = predictor.exploratory_data_processing(df)
        if df_cleaned is not None:
            preprocessor = predictor.preprocess_features(df_cleaned)
            if preprocessor:
                predictor.train_and_evaluate(preprocessor)

    # Optional new predictions:
    # predictor.predict_new_data('NewHouseData.xlsx')
