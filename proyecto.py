import numpy as np
import re

word_spam_probs = {
    'free': -1.0,
    'win': -1.2,
    'money': -1.1,
    'click': -1.3,
    'offer': -1.5,
    'now': -1.6,
    'urgent': -1.4,
    'hello': -3.0,
    'meeting': -3.2,
    'project': -3.1,
    'schedule': -3.3
}

# Probabilidades base (log priors)
log_prob_spam = np.log(0.4)  # 40% spam
log_prob_ham = np.log(0.6)   # 60% ham

def preprocess(text):
    """Limpieza básica: a minúsculas y extrae palabras."""
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def predict_spam(text):
    words = preprocess(text)
    spam_score = log_prob_spam
    ham_score = log_prob_ham
    word_contributions = {}

    # Sumar contribuciones de cada palabra
    for word in words:
        log_p_spam_w = word_spam_probs.get(word, -3.5)   # smoothing
        log_p_ham_w  = -3.5                              # symmetric smoothing
        spam_score += log_p_spam_w
        ham_score  += log_p_ham_w
        word_contributions[word] = log_p_spam_w

    # Calcular probabilidad normalizada de SPAM
    max_score = max(spam_score, ham_score)
    # Para estabilidad numérica
    spam_exp = np.exp(spam_score - max_score)
    ham_exp  = np.exp(ham_score  - max_score)
    p_spam = spam_exp / (spam_exp + ham_exp)

    classification = "SPAM" if p_spam > 0.5 else "HAM"

    # Top 3 palabras con mayor log‑probabilidad de SPAM (más cercanas a 0)
    top_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "input": text,
        "classification": classification,
        "p_spam": p_spam,
        "top_predictive_words": [w for w, _ in top_words]
    }

if __name__ == "__main__":
    prompt = input("Ingresa un mensaje de texto: ")
    result = predict_spam(prompt)
    print(f"\nTexto: \"{result['input']}\"")
    print(f"Clasificación: {result['classification']}")
    print(f"Probabilidad de SPAM: {result['p_spam']:.4f}")
    print("Top 3 palabras más indicativas de SPAM:")
    for word in result['top_predictive_words']:
        print(f" - {word}")
