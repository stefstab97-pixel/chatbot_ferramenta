import streamlit as st
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI

# -----------------------------
# 1️⃣ API Key OpenAI sicura
# -----------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API Key non trovata! Imposta la variabile OPENAI_API_KEY.")
client = OpenAI(api_key=api_key)

# -----------------------------
# 2️⃣ Percorso assoluto dei file
# -----------------------------
cartella = r"C:\Users\stabl\OneDrive\Desktop\chatbot_ferroSTErc0"
faiss_index_path = os.path.join(cartella, "prodotti_index.faiss")
texts_path = os.path.join(cartella, "prodotti_texts.pkl")

# -----------------------------
# 3️⃣ Caricamento FAISS e testi prodotti
# -----------------------------
faiss_index = faiss.read_index(faiss_index_path)

with open(texts_path, "rb") as f:
    prodotti_texts = pickle.load(f)

# -----------------------------
# 4️⃣ Funzione ricerca semantica
# -----------------------------
def cerca_prodotti(query, k=3):
    """
    Restituisce i primi k prodotti più simili semanticamente alla query.
    """
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    query_emb_array = np.array(query_emb, dtype='float32').reshape(1, -1)
    D, I = faiss_index.search(query_emb_array, k)
    
    return [prodotti_texts[i] for i in I[0]]

# -----------------------------
# 5️⃣ Few-shot examples
# -----------------------------
few_shot = [
    # Ferramenta
    {"user": "Voglio un trapano", 
     "assistant": "Ti consiglio un avvitatore XYZ, perfetto per legno, Marca ABC, Prezzo: 39.99"},
    {"user": "Mi serve una chiave inglese", 
     "assistant": "Ti suggerisco la chiave inglese LMN, regolabile da 8 a 24 mm, Marca OPQ, Prezzo: 12.50"},
    
    # Cancelleria
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
# 6️⃣ Streamlit Chatbot UI
# -----------------------------
st.title("🛠️ Chatbot Ferramenta & Cancelleria")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Input utente
user_input = st.text_input("Scrivi la tua richiesta:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Ricerca prodotti più simili
    prodotti_risultati = cerca_prodotti(user_input)
    
    # Few-shot + query utente
    prompt = "Sei un assistente vendita di ferramenta e cancelleria. Rispondi consigliando il prodotto più adatto.\n"
    for ex in few_shot:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    prompt += f"Utente: {user_input}\nAssistente:"

    # Chiamata modello OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    assistant_reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# Mostra chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**Tu:** {msg['content']}")
    else:
        st.markdown(f"**Assistente:** {msg['content']}")

# Mostra prodotti trovati dal database
if user_input:
    st.subheader("💡 Prodotti consigliati dal database:")
    for i, r in enumerate(prodotti_risultati, 1):
        st.write(f"{i}. {r}")
