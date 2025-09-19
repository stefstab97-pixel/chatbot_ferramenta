import streamlit as st
import pickle
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# -----------------------------
# 1️⃣ Carica API Key OpenAI
# -----------------------------
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -----------------------------
# 2️⃣ Carica Vector Store FAISS e testi
# -----------------------------
cartella = "C:/Users/stabl/OneDrive/Desktop/vector_store"
faiss_index = faiss.read_index(os.path.join(cartella, "prodotti_index.faiss"))

with open(os.path.join(cartella, "prodotti_texts.pkl"), "rb") as f:
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
# 4️⃣ Few-shot examples
# -----------------------------
few_shot = [
    {"user": "Ho bisogno di etichette adesive", 
     "assistant": "Etichette adesive Navigator, confezione da 25 pezzi, Prezzo: 27.71€, made in Spagna, codice 602EF"},
    
    {"user": "Mi serve un evidenziatore giallo", 
     "assistant": "Evidenziatore giallo Pilot, confezione da 50 pezzi, Prezzo: 19.35€, provenienza UK, codice 715EF"},
    
    {"user": "Vorrei dei gessetti colorati", 
     "assistant": "Gessetti colorati Navigator, confezione da 20 pezzi, Prezzo: 617.00€, provenienza Germania, codice 204CD"},
    
    {"user": "Cerco cartucce per stampante", 
     "assistant": "Cartucce per stampante Navigator, confezione da 100 pezzi, Prezzo: 37.54€, provenienza Cina, codice 152MN"}
]

# -----------------------------
# 5️⃣ Streamlit UI migliorata
# -----------------------------
st.set_page_config(
    page_title="🛠️ Chatbot Ferramenta & Cancelleria",
    page_icon="🛒",
    layout="wide"
)

st.title("🛠️ Chatbot Ferramenta & Cancelleria")
st.markdown("""
Benvenuto! Scrivi la tua richiesta e ti suggerirò i prodotti più adatti.
Puoi anche selezionare una **categoria** per affinare i risultati.
""")

# Selezione categoria
categorie = ["Tutti", "Cancelleria", "Ferramenta", "Cartucce e Stampanti", "Altro"]
categoria = st.selectbox("Seleziona categoria:", categorie)

# Input dell'utente
user_input = st.text_input("Scrivi la tua richiesta:")

# Pulsante invio
if st.button("Cerca prodotto") and user_input:
    st.info("⏳ Sto cercando i prodotti più adatti...")

    # -----------------------------
    # 6️⃣ Ricerca prodotti
    # -----------------------------
    risultati = cerca_prodotti(user_input)
    
    # Filtra in base alla categoria selezionata (semplice matching su testo)
    if categoria != "Tutti":
        risultati = [r for r in risultati if categoria.lower() in r.lower()]
    
    # -----------------------------
    # 7️⃣ Costruzione prompt few-shot
    # -----------------------------
    prompt = "Sei un assistente vendita di ferramenta e cancelleria. Rispondi consigliando il prodotto più adatto.\n"
    for ex in few_shot:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    prompt += f"Utente: {user_input}\nAssistente:"

    # -----------------------------
    # 8️⃣ Chiamata modello OpenAI
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    # -----------------------------
    # 9️⃣ Visualizzazione risultati UI/UX
    # -----------------------------
    st.subheader("💡 Prodotti consigliati dal vector store:")
    if risultati:
        for i, r in enumerate(risultati, 1):
            st.success(f"{i}. {r}")
    else:
        st.warning("Nessun prodotto trovato per la categoria selezionata.")

    st.subheader("📝 Risposta generata dal modello:")
    st.write(response.choices[0].message.content)

    # Pulsante per info aggiuntive
    if st.button("Mostra informazioni aggiuntive"):
        st.info("ℹ️ Qui puoi aggiungere dettagli come disponibilità in magazzino, alternative o specifiche tecniche.")
        # Esempio di informazioni aggiuntive
        for i, r in enumerate(risultati, 1):
            st.write(f"✅ {r} - Disponibilità: In stock, Sconto attuale: 5%")
