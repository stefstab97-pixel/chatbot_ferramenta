import pandas as pd
import os
from openai import OpenAI
import faiss
import numpy as np
import pickle

# -----------------------------
# 1️⃣ Imposta la tua API key OpenAI
# -----------------------------
# Sostituisci "INSERISCI_LA_TUA_API_KEY" con la tua chiave reale
from dotenv import load_dotenv
import os

# carica le variabili da .env
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# esempio: inizializzazione client OpenAI
from openai import OpenAI
client = OpenAI(api_key=api_key)

# -----------------------------
# 2️⃣ Carica il CSV normalizzato
# -----------------------------
# Sostituisci il percorso con il tuo CSV normalizzato dei prodotti
csv_path = "prodotti_normalizzati.csv"
df = pd.read_csv(csv_path)

# Rimuove eventuali spazi dai nomi delle colonne
df.columns = df.columns.str.strip()

# -----------------------------
# 3️⃣ Crea una colonna "text" combinando le informazioni dei prodotti
# -----------------------------
# Qui puoi aggiungere/rimuovere colonne a piacere
df['text'] = df.apply(
    lambda row: f"{row['Nome_prodotto']} {row['Marca']} {row['Descrizione']} Prezzo: {row['Prezzo_unitario']}", axis=1
)

# -----------------------------
# 4️⃣ Genera gli embeddings dei prodotti
# -----------------------------
embeddings = []

for text in df['text']:
    emb = client.embeddings.create(
        model="text-embedding-3-small",  # Puoi usare anche "text-embedding-3-large" se vuoi più accuratezza
        input=text
    )
    embeddings.append(emb.data[0].embedding)

# -----------------------------
# 5️⃣ Costruisci il vector store FAISS
# -----------------------------
dimension = len(embeddings[0])  # Dimensione dell'embedding
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# -----------------------------
# 6️⃣ Salva l'indice FAISS e i testi associati
# -----------------------------
faiss.write_index(index, "prodotti_index.faiss")

with open("prodotti_texts.pkl", "wb") as f:
    pickle.dump(df['text'].tolist(), f)

print("✅ Embeddings generati e vector store salvato!")
