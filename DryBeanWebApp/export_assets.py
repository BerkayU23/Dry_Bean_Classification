import torch
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from pathlib import Path

# 1. Veri Setini Oku (Bu dosyanın çalışma dizininde Dry_Bean_Dataset.csv olmalıdır)
try:
    df = pd.read_csv("Dry_Bean_Dataset.csv")
except FileNotFoundError:
    print("Hata: 'Dry_Bean_Dataset.csv' dosyası bulunamadı. Lütfen veri setini bu klasöre ekleyin.")
    exit()

df = df.drop_duplicates()

X = df.drop("Class", axis=1).values
y = df["Class"].values

# 2. LabelEncoder'ı fit et ve kaydet
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, "label_encoder.pkl")
print("LabelEncoder başarıyla 'label_encoder.pkl' olarak kaydedildi.")

# 3. Train/Test Split (Sizin kullandığınız aynı random_state=23 değeriyle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# 4. StandardScaler'ı fit et ve kaydet
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")
print("StandardScaler başarıyla 'scaler.pkl' olarak kaydedildi.")

# 5. Model Eğitimi (Opsiyonel) - Eğer model.pth zaten varsa sadece kopyalayın, yoksa bu kod ile yeniden eğitebilirsiniz.
# Bu script, sadece scaler ve encoder'ı kurtarmak/kaydetmek için kısaltılmıştır.
print("Not: Model ağırlık dosyanız ('DryBeanClassifier.pth') eğer elinizde varsa doğrudan bu klasördeki 'DryBeanMC' klasörüne veya kök dizine kopyalayabilirsiniz.")
