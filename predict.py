import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

class LoanStatusPredictor:
    def __init__(self, model_path='xgb_model.pkl', scaler_path='scaler.pkl',
                 encoder_path='encoders.pkl', columns_path='columns.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoder_path)
        self.columns = joblib.load(columns_path)

    def preprocess(self, df):
        df = df.copy()

        # Normalisasi gender
        df['person_gender'] = df['person_gender'].str.lower().replace('fe male', 'female')

        # Imputasi kolom numerik
        if 'person_income' in df.columns:
            df['person_income'] = df['person_income'].fillna(df['person_income'].median())

        # Scaling kolom numerik
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Encoding binary kategorikal
        for col, encoder in self.encoders.items():
            if isinstance(encoder, LabelEncoder):
                if col in df.columns:
                    df[col] = encoder.transform(df[col])

        # Encoding ordinal kategorikal
        for col, encoder in self.encoders.items():
            if isinstance(encoder, OrdinalEncoder):
                if col in df.columns:
                    df[col] = encoder.transform(df[[col]])

        # One-hot encoding
        one_hot_cols = ['loan_intent', 'person_home_ownership']
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        df = df.reindex(columns=self.columns, fill_value=0)

        return df

    def predict(self, raw_df):
        processed_df = self.preprocess(raw_df)
        prediction = self.model.predict(processed_df)
        return prediction

