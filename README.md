# 🫘 Dry Bean Classification with PyTorch & FastAPI

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Bu proje, 7 farklı kuru fasulye türünü (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) yapısal ve boyutsal özelliklerine göre sınıflandırmak için geliştirilmiş uçtan uca bir makine öğrenmesi uygulamasıdır. PyTorch kullanılarak geliştirilen Derin Öğrenme (Deep Learning) modeli, modern ve hızlı bir web framework'ü olan FastAPI ile birleştirilerek gerçek zamanlı tahmin yapabilen bir API servisine dönüştürülmüştür.

## 🚀 Özellikler

* **Veri Ön İşleme:** Dry_Bean_Dataset.csv dosyası üzerinden veriler okunmuş, tekrar eden satırlar temizlenmiş ve sayısal veriler `StandardScaler` ile ölçeklendirilmiştir.
* **Derin Öğrenme Modeli:** PyTorch kullanılarak 16 girişli, 32 nöronlu gizli katmanlara sahip ve 7 çıkışlı tam bağlı (fully connected) bir sinir ağı mimarisi kurulmuştur.
* **Model Mimarisi:** PyTorch ile 16 giriş özellikli, `ReLU` aktivasyon fonksiyonu kullanan 32 nöronlu iki gizli katmana sahip ve 7 çıkışlı tam bağlı (Fully Connected) bir sinir ağı inşa edilmiştir.
* **Performans Takibi:** Modelin doğruluğu (accuracy) ve hata oranı (loss) `TorchMetrics` kütüphanesi ile `MulticlassAccuracy` ve `MulticlassConfusionMatrix` kullanılarak değerlendirilmiştir.
* **API ve Web Arayüzü:** Eğitilen model (`DryBeanClassifier.pth`), tahminleri görselleştirmek için bir web arayüzüne API aracılığıyla bağlanmıştır.

## 🛠️ Kullanılan Teknolojiler

* **Veri İşleme:** Pandas, Scikit-Learn
* **Derin Öğrenme:** PyTorch, TorchMetrics
* **Görselleştirme:** Matplotlib, Seaborn
* **API & Web:** FastAPI

## 📊 Model Mimarisi ve Eğitim Süreci

Model eğitimi sırasında aşağıdaki hiperparametreler kullandım:
* **Epoch Sayısı:** 501
* **Optimizasyon Algoritması:** Adam Optimizer (Learning Rate: 0.003)
* **Kayıp Fonksiyonu:** CrossEntropyLoss

Model eğitimi sonucunda elde edilen hata matrisi (Confusion Matrix) proje içerisinde görselleştirilmiş ve model ağırlıkları `DryBeanMC/DryBeanClassifier.pth` yoluna kaydedilmiştir.

## 📊 Eğitim Süreci ve Model Performansı

Model, **501 epoch** boyunca, `0.003` öğrenme oranıyla (Learning Rate) ve **Adam Optimizer** kullanılarak eğitilmiştir. Eğitim ve test seti değerlendirmelerinde aşağıdaki yüksek başarı oranları elde edilmiştir:

* **Eğitim Doğruluğu (Train Accuracy):** `%94.51`
* **Test Doğruluğu (Test Accuracy):** `%93.46`
* **Test Kaybı (Test Loss):** `0.1852`

* **Not:** Eğitim ve test doğruluğu arasındaki bu yakınlık (~%1 fark), modelin aşırı öğrenmeye (overfitting) düşmediğini ve yeni, görülmemiş verilere karşı yüksek bir genelleme yeteneğine sahip olduğunu kanıtlamaktadır.

* ### 🔍 Karmaşıklık Matrisi (Confusion Matrix) Analizi

Test verisi üzerindeki tahminleri detaylı incelediğimizde modelin sınıflandırma karakteristiği hakkında şu kritik sonuçlara ulaştım:

1. **Kusursuz Tahmin (Sınıf 1):** Model, Sınıf 1 etiketli fasulyeleri %100 başarıyla tahmin etmiştir (92 örneğin 92'si doğru). Bu durum, bu fasulye türünün ayırt edici fiziksel özelliklerinin (muhtemelen boyut olarak Bombay gibi belirgin bir tür olması) model tarafından net bir şekilde yakalandığını gösterir.
2. **Sınıf 3 ve Sınıf 6 Karışıklığı:** Modelin en çok zorlandığı senaryo Sınıf 3 ile Sınıf 6'nın birbirine karıştırılmasıdır. 
   * Sınıf 3'e ait 40 örnek Sınıf 6 olarak tahmin edilmiştir.
   * Sınıf 6'ya ait 43 örnek ise Sınıf 3 olarak tahmin edilmiştir.
   * Bu bulgu, veri setindeki bu iki fasulye türünün geometrik (alan, çevre, vb.) olarak birbirine yapısal anlamda çok benzediğine işaret eder.
3. **Genel Başarı:** Çoğunluk sınıflarından Sınıf 3 (662 doğru), Sınıf 5 (420 doğru) ve Sınıf 6 (475 doğru) tahminlerinde ana köşegen üzerindeki (True Positive) değerlerin çok yüksek olması modelin genel veri yapısını mükemmel kavradığını göstermektedir.

## 💻 Kurulum ve Kullanım

Projeyi kendi bilgisayarında çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu klonlayın:
   git clone 

2. Gerekli kütüphaneleri yükleyin:
   pip install -r "requirements.txt"

3. API sunucusunu başlatın:
   (Buraya API'yi çalıştırma komutunu ekle, örn: python -m uvicorn main:app --reload)

4. Web arayüzüne tarayıcınızdan erişin.
