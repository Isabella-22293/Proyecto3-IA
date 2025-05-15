import numpy as np
import pandas as pd
import re
from collections import defaultdict
from math import log

def preprocess(text):
    """Convierte texto a minúsculas y extrae palabras individuales."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def train_from_csv(file_path):
    df = pd.read_csv(file_path, sep=';', encoding='latin1')

    # Eliminar filas con valores nulos en Label o SMS_TEXT
    df = df.dropna(subset=['Label', 'SMS_TEXT'])

    spam_words = []
    ham_words = []
    total_spam = 0
    total_ham = 0

    for _, row in df.iterrows():
        label = row['Label'].strip().lower()
        words = preprocess(row['SMS_TEXT'])
        if label == 'spam':
            spam_words.extend(words)
            total_spam += 1
        elif label == 'ham':
            ham_words.extend(words)
            total_ham += 1

    spam_counts = defaultdict(int)
    ham_counts = defaultdict(int)

    for word in spam_words:
        spam_counts[word] += 1
    for word in ham_words:
        ham_counts[word] += 1

    all_words = set(spam_counts.keys()) | set(ham_counts.keys())
    vocab_size = len(all_words)

    word_spam_probs = {}
    for word in all_words:
        # Laplace smoothing
        p_w_spam = (spam_counts[word] + 1) / (len(spam_words) + vocab_size)
        p_w_ham = (ham_counts[word] + 1) / (len(ham_words) + vocab_size)
        word_spam_probs[word] = (log(p_w_spam), log(p_w_ham))

    # Priors
    total = total_spam + total_ham
    log_prob_spam = log(total_spam / total)
    log_prob_ham = log(total_ham / total)

    return word_spam_probs, log_prob_spam, log_prob_ham

def predict_spam(text, word_probs, log_prob_spam, log_prob_ham):
    words = preprocess(text)
    spam_score = log_prob_spam
    ham_score = log_prob_ham
    word_contributions = {}

    for word in words:
        log_p_spam_w, log_p_ham_w = word_probs.get(word, (log(1e-6), log(1e-6)))
        spam_score += log_p_spam_w
        ham_score += log_p_ham_w
        word_contributions[word] = log_p_spam_w

    max_score = max(spam_score, ham_score)
    spam_exp = np.exp(spam_score - max_score)
    ham_exp = np.exp(ham_score - max_score)
    p_spam = spam_exp / (spam_exp + ham_exp)
    classification = "SPAM" if p_spam > 0.5 else "HAM"
    top_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "input": text,
        "classification": classification,
        "p_spam": p_spam,
        "top_predictive_words": [w for w, _ in top_words]
    }

if __name__ == "__main__":
    ruta_csv = input("Ingresa la ruta del archivo CSV para entrenamiento: ")
    word_probs, log_spam, log_ham = train_from_csv(ruta_csv)

    while True:
        prompt = input("\nIngresa un mensaje de texto (o 'salir'): ")
        if prompt.lower() == 'salir':
            break
        result = predict_spam(prompt, word_probs, log_spam, log_ham)
        print(f"\nTexto: \"{result['input']}\"")
        print(f"Clasificación: {result['classification']}")
        print(f"Probabilidad de SPAM: {result['p_spam']:.4f}")
        print("Top 3 palabras más indicativas de SPAM:")
        for word in result['top_predictive_words']:
            print(f" - {word}")
