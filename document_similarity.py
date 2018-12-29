import csv
import random
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer




# Eğitim Verilerinin Eklenmesi


train_file = open('./data/train.csv')
egitim_verisi = [row for row in csv.reader(train_file)][1:]  # ilk satırı es geç
random.shuffle(egitim_verisi)
try:
    egitim_girdisi = [row[0].decode('utf8') for row in egitim_verisi]
except AttributeError:
    egitim_girdisi = [row[0] for row in egitim_verisi]
egitim_ciktisi = [row[1] for row in egitim_verisi]

# Test Verilerinin Eklenmesi

test_file = open('./data/test.csv')
test_rows = [row for row in csv.reader(test_file)][1:]  # ilk satırı es geç
try:
    test_girdisi = [row[0].decode('utf8') for row in test_rows]
except AttributeError:  # it's python 3
    test_girdisi = [row[0] for row in test_rows]
test_ciktisi = [row[1] for row in test_rows]

# Model Sınıflandırıcı tanımlama

#pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
#                        ('classifier', LogisticRegression())])

pipeline_naive_bayes = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
                        ('classifier', MultinomialNB())])


# Model eğitim ve test aşaması

for train_size in (100, 500, 1000, 2000, 5000,len(egitim_girdisi)):

    print('--------------------------------------')

    # tfidf
    pipeline_naive_bayes.fit(egitim_girdisi[:train_size], egitim_ciktisi[:train_size])   #egitim kısmı
    print("{}.adım  Test sonucu : {}".format(train_size,pipeline_naive_bayes.score(test_girdisi, test_ciktisi)))


x = open("dosya.txt","r")

b = []  #tüm çıktıları diziye atarak sonrasında yüzde hesabı yapabiliriz.
for i in x.readlines():
    a = []
    a.append(i)
    b.append(pipeline_naive_bayes.predict(a)[0])  #Tahmin sonucu

sozluk = {}   #Sözlükte hangi dökümandan kaç tane olduğunu tutuyoruz

for i in b:
    sozluk[i] = b.count(i)
print(sozluk)
for i,k in enumerate(sozluk) :
    print("yüzde %{} D{}.txt".format(sozluk[k]/len(b)*100,k))

#örnek cümle gönderimi aşağıdaki gibidir.
#print(pipeline_naive_bayes.predict(["rüyalar alanından bu yana tiyatroları vurmak için beyzbol hakkında en iyi film"]))
