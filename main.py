"""
Makine Öğrenmesi Ara Ödevi
--------------------------
Amaç:
    Sentetik müşteri verileri kullanarak müşteri ayrılma (churn) tahmini için
    temel bir sınıflandırma akışı kurmak.

Kullanılan kütüphaneler:
    pandas, numpy, matplotlib, scikit-learn

Çalıştırma:
    1. Gerekli paketleri yükleyin:
       pip install -r requirements.txt
    2. Dosyayı çalıştırın:
       python main.py

Dosya; veri oluşturma, veri inceleme, eksik değer kontrolü, öznitelik üretme,
train-validation-test ayırma, ön işleme, model eğitimi, model karşılaştırma
ve test değerlendirmesi adımlarını içerir.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
DATA_PATH = Path("musteri_ayrilma_verisi.csv")


def sigmoid(x):
    """Bir değeri 0 ile 1 arasında olasılığa dönüştürür."""
    return 1 / (1 + np.exp(-x))


def veri_seti_olustur(n_satir=500, random_state=RANDOM_STATE):
    """En az 100 satırlık sentetik müşteri ayrılma veri seti oluşturur."""
    rng = np.random.default_rng(random_state)

    yas = rng.integers(18, 71, n_satir)
    gelir = np.clip(
        rng.normal(loc=52_000, scale=18_000, size=n_satir),
        15_000,
        120_000,
    ).round(0)
    abonelik_suresi = rng.integers(1, 121, n_satir)
    destek_talebi_sayisi = np.clip(rng.poisson(2.0, n_satir), 0, 9)
    aylik_kullanim_saati = np.clip(
        rng.normal(loc=42, scale=18, size=n_satir),
        3,
        120,
    ).round(1)

    sehir = rng.choice(
        ["Ankara", "İstanbul", "İzmir", "Bursa", "Antalya"],
        size=n_satir,
        p=[0.22, 0.32, 0.18, 0.15, 0.13],
    )
    uyelik_tipi = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n_satir,
        p=[0.42, 0.38, 0.20],
    )

    churn_skoru = (
        -2.0
        + 0.70 * (destek_talebi_sayisi >= 3)
        + 1.10 * (destek_talebi_sayisi >= 5)
        + 1.30 * (abonelik_suresi < 12)
        + 0.70 * ((abonelik_suresi >= 12) & (abonelik_suresi < 24))
        + 0.90 * (uyelik_tipi == "Basic")
        - 0.80 * (uyelik_tipi == "Premium")
        + 0.70 * (gelir < 35_000)
        - 0.40 * (gelir > 75_000)
        + 0.70 * (aylik_kullanim_saati < 20)
        + 0.30 * (yas < 25)
        + rng.normal(0, 0.25, n_satir)
    )

    churn_olasiligi = sigmoid(churn_skoru)
    churn = rng.binomial(1, churn_olasiligi)

    veri = pd.DataFrame(
        {
            "yas": yas,
            "gelir": gelir,
            "abonelik_suresi": abonelik_suresi,
            "destek_talebi_sayisi": destek_talebi_sayisi,
            "aylik_kullanim_saati": aylik_kullanim_saati,
            "sehir": sehir,
            "uyelik_tipi": uyelik_tipi,
            "churn": churn,
        }
    )

    eksik_oranlari = {
        "gelir": 0.05,
        "aylik_kullanim_saati": 0.03,
        "sehir": 0.04,
    }

    for sutun, oran in eksik_oranlari.items():
        adet = int(n_satir * oran)
        indeksler = rng.choice(veri.index, size=adet, replace=False)
        veri.loc[indeksler, sutun] = np.nan

    return veri


def ozellik_uret(veri):
    """Mevcut sütunlardan üç yeni ve anlamlı öznitelik üretir."""
    veri = veri.copy()

    veri["gelir_grubu"] = pd.cut(
        veri["gelir"],
        bins=[0, 35_000, 70_000, np.inf],
        labels=["Düşük", "Orta", "Yüksek"],
    ).astype("object")

    veri["destek_talebi_var_mi"] = np.where(
        veri["destek_talebi_sayisi"] > 0,
        "Evet",
        "Hayır",
    )

    veri["abonelik_yili"] = (veri["abonelik_suresi"] / 12).round(1)

    return veri


def metrikleri_hesapla(y_gercek, y_tahmin):
    """Sınıflandırma metriklerini sözlük olarak döndürür."""
    return {
        "accuracy": accuracy_score(y_gercek, y_tahmin),
        "precision": precision_score(y_gercek, y_tahmin, zero_division=0),
        "recall": recall_score(y_gercek, y_tahmin, zero_division=0),
        "f1_score": f1_score(y_gercek, y_tahmin, zero_division=0),
    }


def main():
    veri = veri_seti_olustur(n_satir=500)
    veri.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Veri seti oluşturuldu ve kaydedildi: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    print("\nİLK 5 SATIR")
    print(df.head())

    print("\nSATIR-SÜTUN SAYISI")
    print(df.shape)

    print("\nHEDEF DEĞİŞKEN DAĞILIMI - ADET")
    print(df["churn"].value_counts().sort_index())

    print("\nHEDEF DEĞİŞKEN DAĞILIMI - ORAN")
    print(df["churn"].value_counts(normalize=True).sort_index().round(3))

    print("\nEKSİK DEĞER SAYILARI")
    print(df.isnull().sum())

    df = ozellik_uret(df)
    print("\nYENİ ÖZNİTELİKLERDEN ÖRNEKLER")
    print(
        df[
            [
                "gelir",
                "gelir_grubu",
                "destek_talebi_sayisi",
                "destek_talebi_var_mi",
                "abonelik_suresi",
                "abonelik_yili",
            ]
        ].head()
    )

    X = df.drop(columns="churn")
    y = df["churn"]

    X_train, X_gecici, y_train, y_gecici = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_gecici,
        y_gecici,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_gecici,
    )

    print("\nVERİ BÖLÜMLERİ")
    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_validation.shape}")
    print(f"Test       : {X_test.shape}")

    sayisal_sutunlar = X.select_dtypes(include=np.number).columns.tolist()
    kategorik_sutunlar = X.select_dtypes(exclude=np.number).columns.tolist()

    print("\nSAYISAL SÜTUNLAR")
    print(sayisal_sutunlar)

    print("\nKATEGORİK SÜTUNLAR")
    print(kategorik_sutunlar)

    sayisal_pipeline = Pipeline(
        steps=[
            ("eksik_doldurma", SimpleImputer(strategy="median")),
            ("olcekleme", StandardScaler()),
        ]
    )

    kategorik_pipeline = Pipeline(
        steps=[
            ("eksik_doldurma", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    on_isleme = ColumnTransformer(
        transformers=[
            ("sayisal", sayisal_pipeline, sayisal_sutunlar),
            ("kategorik", kategorik_pipeline, kategorik_sutunlar),
        ]
    )

    modeller = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    validation_sonuclari = []
    egitilmis_pipeline_lari = {}

    for model_adi, model in modeller.items():
        model_pipeline = Pipeline(
            steps=[
                ("on_isleme", on_isleme),
                ("model", model),
            ]
        )

        model_pipeline.fit(X_train, y_train)
        validation_tahmini = model_pipeline.predict(X_validation)

        metrikler = metrikleri_hesapla(y_validation, validation_tahmini)
        metrikler["model"] = model_adi
        validation_sonuclari.append(metrikler)
        egitilmis_pipeline_lari[model_adi] = model_pipeline

    validation_df = (
        pd.DataFrame(validation_sonuclari)
        .set_index("model")
        .sort_values("f1_score", ascending=False)
    )

    print("\nVALIDATION MODEL KARŞILAŞTIRMASI")
    print(validation_df.round(3))

    en_iyi_model_adi = validation_df.index[0]
    print(f"\nSeçilen model: {en_iyi_model_adi}")

    X_train_final = pd.concat([X_train, X_validation], axis=0)
    y_train_final = pd.concat([y_train, y_validation], axis=0)

    en_iyi_pipeline = clone(egitilmis_pipeline_lari[en_iyi_model_adi])
    en_iyi_pipeline.fit(X_train_final, y_train_final)

    test_tahmini = en_iyi_pipeline.predict(X_test)
    test_metrikleri = metrikleri_hesapla(y_test, test_tahmini)

    print("\nTEST METRİKLERİ")
    for metrik_adi, deger in test_metrikleri.items():
        print(f"{metrik_adi:10s}: {deger:.3f}")

    cm = confusion_matrix(y_test, test_tahmini)
    print("\nCONFUSION MATRIX")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Kalır (0)", "Ayrılır (1)"],
    ).plot(values_format="d")
    plt.title(f"Confusion Matrix - {en_iyi_model_adi}")
    plt.tight_layout()
    plt.show()

    print("\nKISA SONUÇ YORUMU")
    print(
        f"Validation F1-score değerine göre en iyi model "
        f"{en_iyi_model_adi} oldu. Bu modelin test F1-score değeri "
        f"{test_metrikleri['f1_score']:.3f} olarak hesaplandı. "
        "Logistic Regression doğrusal ilişkileri iyi yakalayabilir; "
        "Decision Tree değişkenler arasındaki eşik tabanlı ilişkileri öğrenebilir; "
        "KNN ise ölçeklemeye ve komşu sayısına daha duyarlıdır. "
        "Sentetik veri setinde churn davranışı belirli kurallarla üretildiği için "
        "bu kuralları daha iyi yakalayan model daha yüksek performans göstermiştir."
    )


if __name__ == "__main__":
    main()
