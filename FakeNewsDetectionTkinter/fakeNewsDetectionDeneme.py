import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Veri yükleme ve ön işleme
news = pd.read_csv('news.csv')
news_copy = news.copy()
news = news[news['text'].str.strip().ne('')]
news = news.dropna(subset=['text'])
news = news.head(3000)
news.drop_duplicates(inplace=True)
news = news.drop(['title'], axis=1)
news.drop("Unnamed: 0", axis=1, inplace=True)

def word_operations(text):
    # Küçük harfe dönüştürülmesi
    text = text.lower()
    # Linklerin silinmesi
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # HTML Taglerinin silinmesi
    text = re.sub(r'<.*?>', ' ', text)
    # Noktalama işaretlerinin silinmesi
    text = re.sub(r'[^\w\s]', ' ', text)
    # Sayıların silinmesi
    text = re.sub(r'\d', ' ', text)
    # Yeni satır karakterlerinin silinmesi (\n)
    text = re.sub(r'\n', ' ', text)
    # Sekme karakterlerinin silinmesi (\t)
    text = re.sub(r'\t', ' ', text)
    # Fazla boşlukların silinmesi
    text = re.sub(r'  ', ' ', text)
    # Özel karakterlerin silinmesi
    text = re.sub(r'[!@#$%^&*()_+-={}[]|:;"\'<>,.?/~\]', ' ', text)
    return text

def remove_duplicate_words(text):
    words = text.split()
    unique_words = list(dict.fromkeys(words))
    return ' '.join(unique_words)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

x = news['text']
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

vectorizations = {
    "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
    "Bag of Words": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
}

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear', gamma="auto"),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=0, activation="logistic"),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric='cosine')
}

# Model eğitimi ve değerlendirmesi
def train_and_evaluate(model_name, vectorization_method):
    # Öznitelik çıkarımı seçimi
    vectorizer = vectorizations[vectorization_method]
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    
    model = models[model_name]
    model.fit(xv_train, y_train)
    predictions = model.predict(xv_test)
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1_score": f1_score(y_test, predictions, average='weighted')
    }
    return metrics, vectorizer

def predict_news(news_text, model_name, vectorizer):
    new_test = pd.DataFrame({"text": [news_text]})
    new_test["text"] = new_test["text"].apply(word_operations)
    new_test["text"] = new_test["text"].apply(remove_stop_words)
    new_test["text"] = new_test["text"].apply(lemmatize_text)
    new_xv_test = vectorizer.transform(new_test["text"])
    model = models[model_name]
    prediction = model.predict(new_xv_test)
    return "Sahte" if prediction[0] == 0 else "Gerçek"

# Tkinter GUI
def create_gui():
    root = tk.Tk()
    root.title("Sahte Haber Tespiti")
    root.geometry("700x900")
    root.configure(bg="#f5f5f5")

    title_label = tk.Label(root, text="Sahte Haber Tespiti", font=("Arial", 32, "bold"), bg="#f5f5f5", fg="#333")
    title_label.pack(pady=10)

    tk.Label(root, text="Öznitelik Çıkarımı Yöntemi Seçiniz:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w", padx=20, pady=5)
    feature_var = tk.StringVar(value="TF-IDF")
    for vectorization in vectorizations.keys():
        ttk.Radiobutton(root, text=vectorization, variable=feature_var, value=vectorization).pack(anchor="w", padx=40)

    tk.Label(root, text="Makine Öğrenmesi Modeli Seçiniz:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w", padx=20, pady=5)
    model_var = tk.StringVar(value="Logistic Regression")
    for model in models.keys():
        ttk.Radiobutton(root, text=model, variable=model_var, value=model).pack(anchor="w", padx=40)

    parameter_frame = tk.Frame(root, bg="#f5f5f5")
    parameter_frame.pack(pady=10, fill="x", padx=20)
    tk.Label(parameter_frame, text="Model Parametreleri:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w")

    knn_frame = tk.Frame(parameter_frame, bg="#f5f5f5")
    tk.Label(knn_frame, text="KNN - K Değeri:", bg="#f5f5f5").pack(side="left", padx=5)
    knn_k_var = tk.IntVar(value=5)
    ttk.Entry(knn_frame, textvariable=knn_k_var, width=5).pack(side="left", padx=5)
    tk.Label(knn_frame, text="Mesafe Ölçüm Metriği:", bg="#f5f5f5").pack(side="left", padx=5)
    knn_metric_var = tk.StringVar(value="cosine")
    ttk.Entry(knn_frame, textvariable=knn_metric_var, width=10).pack(side="left", padx=5)
    knn_frame.pack(anchor="w", padx=40, pady=5)

    nn_frame = tk.Frame(parameter_frame, bg="#f5f5f5")
    tk.Label(nn_frame, text="Yapay Sinir Ağı - Gizli Katman Sayısı:", bg="#f5f5f5").pack(side="left", padx=5)
    nn_hidden_var = tk.StringVar(value="100")
    ttk.Entry(nn_frame, textvariable=nn_hidden_var, width=10).pack(side="left", padx=5)
    tk.Label(nn_frame, text="Aktivasyon Fonksiyonu:", bg="#f5f5f5").pack(side="left", padx=5)
    nn_activation_var = tk.StringVar(value="logistic")
    ttk.Entry(nn_frame, textvariable=nn_activation_var, width=10).pack(side="left", padx=5)
    nn_frame.pack(anchor="w", padx=40, pady=5)

    svm_frame = tk.Frame(parameter_frame, bg="#f5f5f5")
    tk.Label(svm_frame, text="SVM - Kernel:", bg="#f5f5f5").pack(side="left", padx=5)
    svm_kernel_var = tk.StringVar(value="linear")
    ttk.Entry(svm_frame, textvariable=svm_kernel_var, width=10).pack(side="left", padx=5)
    svm_frame.pack(anchor="w", padx=40, pady=5)

    results_frame = tk.Frame(root, bg="#f5f5f5")
    results_frame.pack(fill="x", padx=20)
    tk.Label(results_frame, text="Sonuçlar:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w")
    results_text = tk.Text(results_frame, height=10, state="disabled", bg="#fff", fg="#333", font=("Arial", 10))
    results_text.pack(fill="both", expand=True, pady=5)

    def update_results(metrics):
        results_text.config(state="normal")
        results_text.delete("1.0", "end")
        results_text.insert("end", f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
        results_text.insert("end", f"Accuracy: {metrics['accuracy']:.2f}\n")
        results_text.insert("end", f"Precision: {metrics['precision']:.2f}\n")
        results_text.insert("end", f"Recall: {metrics['recall']:.2f}\n")
        results_text.insert("end", f"F1 Score: {metrics['f1_score']:.2f}\n")
        results_text.config(state="disabled")

    def train_model():
        model_name = model_var.get()
        vectorization_method = feature_var.get()

        # Parametrelerin uygulanması
        if model_name == "KNN":
            models["KNN"] = KNeighborsClassifier(n_neighbors=knn_k_var.get(), metric=knn_metric_var.get())
        elif model_name == "Neural Network":
            hidden_layer_sizes = tuple(map(int, nn_hidden_var.get().split(",")))
            models["Neural Network"] = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=300, random_state=0,
                                                     activation=nn_activation_var.get())
        elif model_name == "SVM":
            models["SVM"] = SVC(kernel=svm_kernel_var.get(), gamma="auto")

        metrics, vectorizer = train_and_evaluate(model_name, vectorization_method)
        global selected_vectorizer  # Öznitelik çıkarım yöntemini global olarak sakla
        selected_vectorizer = vectorizer
        update_results(metrics)

    train_button = ttk.Button(root, text="Modeli Eğit", command=train_model)
    train_button.pack(pady=10)

    tk.Label(root, text="Tahmin Edilecek Haberi Gir:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w", padx=20, pady=5)
    news_entry = ttk.Entry(root, width=70)
    news_entry.pack(padx=20)

    def predict():
        news_text = news_entry.get()
        model_name = model_var.get()
        if not news_text.strip():
            messagebox.showerror("Hata", "Lütfen tahmin edilecek haberi giriniz.")
            return
        prediction = predict_news(news_text, model_name, selected_vectorizer)
        messagebox.showinfo("Tahmin", f"Girilen haber: {prediction}")

    predict_button = ttk.Button(root, text="Tahmin Et", command=predict)
    predict_button.pack(pady=10)

    root.mainloop()


create_gui()

