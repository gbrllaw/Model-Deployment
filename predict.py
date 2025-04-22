import pandas as pd
import joblib

class LoanStatusPredictor:
    def __init__(self, model_path='xgb_model.pkl', scaler_path='scaler.pkl',
                 encoder_path='encoders.pkl', columns_path='columns.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoder_path)
        self.columns = joblib.load(columns_path)

        self.ordinal_encoders = {
            'person_education': OrdinalEncoder(categories=[['High School', 'Bachelors', 'Masters', 'PhD']])
        }

    def preprocess(self, df):
        df = df.copy()
        df['person_gender'] = df['person_gender'].str.lower().replace('fe male', 'female')
        df['person_income'] = df['person_income'].fillna(df['person_income'].median())

        # Imputasi untuk numeric lainnya kalau ada (bisa kamu expand kalau perlu)

        # Scaling numerik
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Label encoding untuk binary
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Apply Ordinal Encoding to ordinal categorical columns
        for col, encoder in self.ordinal_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[[col]])

        # One-hot encoding (manual karena tidak pakai OneHotEncoder sklearn)
        one_hot_cols = ['loan_intent', 'person_home_ownership']
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        df = df.reindex(columns=self.columns, fill_value=0)

        return df

    def predict(self, raw_df):
        processed_df = self.preprocess(raw_df)
        return self.model.predict(processed_df)

