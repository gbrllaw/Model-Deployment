import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
np.seterr(all='ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import pickle


class LoanXGBoostModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.columns = None
        self.encoders = {}

    def load_and_clean_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['person_gender'] = self.df['person_gender'].str.lower()
        self.df['person_gender'] = self.df['person_gender'].replace('fe male', 'female')

        input_df = self.df.drop('loan_status', axis=1)
        output_df = self.df['loan_status']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            input_df, output_df, test_size=0.2, random_state=42
        )
        # Cek tipe data kolom 'person_income' di x_train
        print(self.x_train['person_income'].dtype)  # Untuk memeriksa tipe data


    def preprocess_data(self):
        # Imputasi nilai kosong
        imputer = SimpleImputer(strategy='median')
        self.x_train['person_income'] = imputer.fit_transform(self.x_train[['person_income']])
        self.x_test['person_income'] = imputer.transform(self.x_test[['person_income']])
        # Verifikasi setelah imputasi
        print(self.x_train.isnull().sum())  # Cek apakah ada missing value di x_train
        print(self.x_test.isnull().sum())   # Cek apakah ada missing value di x_test


        # Scaling numerik
        numeric_cols = self.x_train.select_dtypes(include=['int64', 'float64']).columns
        scaler = RobustScaler()
        self.x_train[numeric_cols] = scaler.fit_transform(self.x_train[numeric_cols])
        self.x_test[numeric_cols] = scaler.transform(self.x_test[numeric_cols])
        self.scaler = scaler  # simpan scaler

        # Encoding binary kategorikal
        label_cols = ['person_gender', 'previous_loan_defaults_on_file']
        for col in label_cols:
            encoder = LabelEncoder()
            self.x_train[col] = encoder.fit_transform(self.x_train[col])
            self.x_test[col] = encoder.transform(self.x_test[col])
            self.encoders[col] = encoder  # simpan encoder

        # Menggunakan OrdinalEncoder untuk kolom yang terurut
        ordinal_cols = ['person_education']  
        ordinal_encoder = OrdinalEncoder()
        self.x_train[ordinal_cols] = ordinal_encoder.fit_transform(self.x_train[ordinal_cols])
        self.x_test[ordinal_cols] = ordinal_encoder.transform(self.x_test[ordinal_cols])
        self.encoders['person_education'] = ordinal_encoder  


        # One-hot encoding untuk kolom multikategori
        one_hot_cols = ['loan_intent', 'person_home_ownership']
        self.x_train = pd.get_dummies(self.x_train, columns=one_hot_cols, drop_first=True)
        self.x_test = pd.get_dummies(self.x_test, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        self.x_train, self.x_test = self.x_train.align(self.x_test, join='left', axis=1, fill_value=0)

        self.columns = self.x_train.columns.tolist()  # simpan struktur kolom

    def train_best_model(self):
        # Hitung class weight
        weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        # Hyperparameter grid
        parameters = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
        }

        xgb_clf = xgb.XGBClassifier(
            scale_pos_weight=weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        # Grid search
        grid = GridSearchCV(xgb_clf, parameters, scoring='f1', cv=5)
        grid.fit(self.x_train, self.y_train)

        best_params = grid.best_params_
        print("Best Parameters:", best_params)

        # Train model terbaik
        self.model = xgb.XGBClassifier(
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            subsample=best_params['subsample'],
            gamma=best_params['gamma'],
            scale_pos_weight=weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        print("\nClassification Report\n")
        print(classification_report(self.y_test, y_pred, target_names=['0', '1']))


    def save_model(self, model_path='xgb_model.pkl', scaler_path='scaler.pkl',
                   columns_path='columns.pkl', encoder_path='encoders.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.columns, columns_path)
        joblib.dump(self.encoders, encoder_path)  # simpan semua encoder
        print("Model, scaler, encoder, and column structure saved successfully.")

if __name__ == "__main__":
    model = LoanXGBoostModel(data_path='Dataset_A_loan.csv')  
    model.load_and_clean_data()
    model.preprocess_data()
    model.train_best_model()
    model.evaluate_model()
    model.save_model()


