# Dry Bean Classification with PyTorch & API

Bu proje, farklı kuru fasulye türlerini boyut ve şekil özelliklerine göre sınıflandırmak için geliştirilmiş uçtan uca (end-to-end) bir makine öğrenmesi uygulamasıdır. PyTorch ile geliştirilen derin öğrenme modeli, eğitildikten sonra bir API ve Web Arayüzü üzerinden kullanıcıların tahmin yapmasına olanak tanır.

## 🚀 Özellikler

* **Veri Ön İşleme:** Dry_Bean_Dataset.csv dosyası üzerinden veriler okunmuş, tekrar eden satırlar temizlenmiş ve sayısal veriler `StandardScaler` ile ölçeklendirilmiştir.
* **Derin Öğrenme Modeli:** PyTorch kullanılarak 16 girişli, 32 nöronlu gizli katmanlara sahip ve 7 çıkışlı tam bağlı (fully connected) bir sinir ağı mimarisi kurulmuştur.
* **Performans Takibi:** Modelin doğruluğu (accuracy) ve hata oranı (loss) `TorchMetrics` kütüphanesi ile `MulticlassAccuracy` ve `MulticlassConfusionMatrix` kullanılarak değerlendirilmiştir[cite: 1].
* **API ve Web Arayüzü:** Eğitilen model (`DryBeanClassifier.pth`), tahminleri görselleştirmek için bir web arayüzüne API aracılığıyla bağlanmıştır[cite: 1].

## 🛠️ Kullanılan Teknolojiler

* **Veri İşleme:** Pandas, Scikit-Learn[cite: 1]
* **Derin Öğrenme:** PyTorch, TorchMetrics[cite: 1]
* **Görselleştirme:** Matplotlib, Seaborn[cite: 1]
* **API & Web:** FastAPI

## 📊 Model Mimarisi ve Eğitim Süreci

Model eğitimi sırasında aşağıdaki hiperparametreler kullandım:
* **Epoch Sayısı:** 501[cite: 1]
* **Optimizasyon Algoritması:** Adam Optimizer (Learning Rate: 0.003)[cite: 1]
* **Kayıp Fonksiyonu:** CrossEntropyLoss[cite: 1]

Model eğitimi sonucunda elde edilen hata matrisi (Confusion Matrix) proje içerisinde görselleştirilmiş ve model ağırlıkları `DryBeanMC/DryBeanClassifier.pth` yoluna kaydedilmiştir[cite: 1].

## 💻 Kurulum ve Kullanım

Projeyi kendi bilgisayarında çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu klonlayın:
   git clone 

2. Gerekli kütüphaneleri yükleyin:
   pip install -r "requirements.txt"

3. API sunucusunu başlatın:
   (Buraya API'yi çalıştırma komutunu ekle, örn: python -m uvicorn main:app --reload)

4. Web arayüzüne tarayıcınızdan erişin.
