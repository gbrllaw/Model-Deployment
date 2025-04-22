import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

class ModelInference:
    def __init__(self, model_path='xgb_model.pkl', scaler_path='scaler.pkl',
                 columns_path='columns.pkl', label_encoder_path='label_encoder.pkl', ordinal_encoder_path='ordinal_encoder.pkl'):
        # Load model, scaler, and encoders
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.columns = joblib.load(columns_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.ordinal_encoder = joblib.load(ordinal_encoder_path)

    def preprocess(self, raw_df):
        df = raw_df.copy()

        # Lowercase dan fix gender typo
        if 'person_gender' in df.columns:
            df['person_gender'] = df['person_gender'].str.lower()
            df['person_gender'] = df['person_gender'].replace('fe male', 'female')

        # Imputasi jika ada kolom kosong (opsional)
        if df['person_income'].isnull().sum() > 0:
            median_income = df['person_income'].median()
            df['person_income'] = df['person_income'].fillna(median_income)

        # Scaling kolom numerik
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Label Encoding untuk kolom kategorikal
        if 'person_gender' in df.columns:
            df['person_gender'] = self.label_encoder.transform(df['person_gender'])

        if 'previous_loan_defaults_on_file' in df.columns:
            df['previous_loan_defaults_on_file'] = self.label_encoder.transform(df['previous_loan_defaults_on_file'])

        # Ordinal Encoding untuk kolom pendidikan
        if 'person_education' in df.columns:
            df['person_education'] = self.ordinal_encoder.transform(df[['person_education']]).flatten()

        # One-hot encoding untuk kolom multikategori
        one_hot_cols = ['loan_intent', 'person_home_ownership']
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

        # Align dengan kolom training (supaya kolom sama)
        df = df.reindex(columns=self.columns, fill_value=0)

        return df

    def predict(self, raw_df):
        processed_df = self.preprocess(raw_df)
        prediction = self.model.predict(processed_df)
        return prediction

