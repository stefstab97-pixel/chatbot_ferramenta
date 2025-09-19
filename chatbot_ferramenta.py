import streamlit as st
import pickle
import faiss
import numpy as np
from openai import OpenAI
import os

cartella = "C:/Users/stabl/OneDrive/Desktop/vector_store"  # cartella dove ci sono i file
faiss_index = faiss.read_index(os.path.join(cartella, "prodotti_index.faiss"))

with open(os.path.join(cartella, "prodotti_texts.pkl"), "rb") as f:
    prodotti_texts = pickle.load(f)


# -----------------------------
# 1️⃣ Imposta API Key OpenAI
# -----------------------------
from dotenv import load_dotenv
import os

# carica le variabili da .env
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# esempio: inizializzazione client OpenAI
from openai import OpenAI
client = OpenAI(api_key=api_key)

# -----------------------------
# 2️⃣ Carica Vector Store FAISS e testi
# -----------------------------
faiss_index = faiss.read_index("prodotti_index.faiss")

with open("prodotti_texts.pkl", "rb") as f:
    prodotti_texts = pickle.load(f)

# -----------------------------
# 3️⃣ Funzione di ricerca semantica
# -----------------------------
def cerca_prodotti(query, k=3):
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    D, I = faiss_index.search(np.array([query_emb]).astype('float32'), k)
    risultati = [prodotti_texts[i] for i in I[0]]
    return risultati

# -----------------------------
# 4️⃣ Few-shot examples basati sui tuoi prodotti
# -----------------------------
few_shot = [
    {"user": "Ho bisogno di etichette adesive", 
     "assistant": "Ti consiglio le etichette adesive Navigator, confezione da 25 pezzi, Prezzo: 27.71€, made in Spagna, codice 602EF"},
    
    {"user": "Mi serve un evidenziatore giallo", 
     "assistant": "Ti suggerisco l'evidenziatore giallo Pilot, confezione da 50 pezzi, Prezzo: 19.35€, provenienza UK, codice 715EF"},
    
    {"user": "Vorrei dei gessetti colorati", 
     "assistant": "Ti consiglio i gessetti colorati Navigator, confezione da 20 pezzi, Prezzo: 617.00€, provenienza Germania, codice 204CD"},
    
    {"user": "Cerco cartucce per stampante", 
     "assistant": "Ti suggerisco le cartucce per stampante Navigator, confezione da 100 pezzi, Prezzo: 37.54€, provenienza Cina, codice 152MN"}
]

# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.title("🛠️ Chatbot Ferramenta & Cancelleria")

user_input = st.text_input("Scrivi la tua richiesta:")

if user_input:
    st.write("⏳ Sto cercando i prodotti più adatti...")
    
    # cerca prodotti simili nel vector store
    risultati = cerca_prodotti(user_input)
    
    # crea prompt few-shot + query utente
    prompt = "Sei un assistente vendita di ferramenta e cancelleria. Rispondi consigliando il prodotto più adatto.\n"
    for ex in few_shot:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    
    prompt += f"Utente: {user_input}\nAssistente:"

    # genera risposta modello
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Mostra i prodotti trovati dal vector store
    st.subheader("💡 Prodotti consigliati:")
    for i, r in enumerate(risultati, 1):
        st.write(f"{i}. {r}")

    # Mostra la risposta generata dal modello
    st.subheader("📝 Risposta modello:")
    st.write(response.choices[0].message.content)
