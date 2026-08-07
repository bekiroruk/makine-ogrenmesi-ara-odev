# Makine Öğrenmesi Ara Ödevi

## Projenin Amacı

Bu projede sentetik olarak oluşturulan müşteri verileri kullanılarak müşterinin
ayrılıp ayrılmayacağını (`churn`) tahmin eden temel bir makine öğrenmesi akışı
uygulanmıştır.

Proje şu adımları içerir:

- Sentetik müşteri veri seti oluşturma
- Temel veri inceleme
- Eksik değer kontrolü ve doldurma
- Öznitelik üretme
- Train-validation-test bölme
- One-Hot Encoding
- Sayısal değişkenleri ölçekleme
- Logistic Regression, KNN ve Decision Tree modellerini eğitme
- Validation F1-score değerlerine göre model seçme
- Seçilen modeli test setinde değerlendirme
- Confusion matrix, accuracy, precision, recall ve F1-score hesaplama

## Veri Seti

Veri seti Python içinde oluşturulmaktadır ve toplam 500 satır içermektedir.

Kullanılan temel sütunlar:

- `yas`
- `gelir`
- `abonelik_suresi`
- `destek_talebi_sayisi`
- `aylik_kullanim_saati`
- `sehir`
- `uyelik_tipi`
- `churn`

Üretilen yeni öznitelikler:

- `gelir_grubu`
- `destek_talebi_var_mi`
- `abonelik_yili`

Hedef değişken:

- `0`: Müşteri kalır
- `1`: Müşteri ayrılır

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

Program çalıştığında:

1. `musteri_ayrilma_verisi.csv` dosyasını oluşturur.
2. Veri inceleme sonuçlarını yazdırır.
3. Modellerin validation performanslarını karşılaştırır.
4. En iyi modeli test verisi üzerinde değerlendirir.
5. Confusion matrix grafiğini gösterir.

## Kısa Sonuç Yorumu

Modeller validation kümesindeki F1-score değerine göre karşılaştırılmaktadır.
F1-score, precision ve recall değerlerini birlikte değerlendirdiği için churn
gibi sınıf dağılımının dengeli olmayabileceği problemlerde uygun bir seçim
ölçütüdür.

Sentetik veri üretiminde kısa abonelik süresi, fazla destek talebi, düşük kullanım,
düşük gelir ve Basic üyelik gibi özellikler churn olasılığını artıracak şekilde
tasarlanmıştır. Bu nedenle bu ilişkileri daha iyi öğrenen model daha yüksek
validation ve test performansı göstermektedir.

> Not: Çalıştırma sonucunda elde edilen kesin metrikler, kullanılan yazılım
> sürümlerine bağlı olarak çok küçük farklılıklar gösterebilir.
