import streamlit as st
import pickle
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import gdown


# -----------------------------
# Funzione per scaricare file da Google Drive usando gdown
# -----------------------------
def scarica_file_gdrive(file_id, percorso_locale):
    if not os.path.exists(percorso_locale):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Scarico {percorso_locale} da Google Drive...")
        gdown.download(url, percorso_locale, quiet=False)
        st.success("Download completato.")


# -----------------------------
# 1️⃣ Carica API Key OpenAI
# -----------------------------
if os.path.exists(".env"):
    load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key non trovata! Imposta OPENAI_API_KEY nel file .env o nelle Secrets su Streamlit Cloud")

client = OpenAI(api_key=api_key)

# -----------------------------
# 2️⃣ Preparazione percorsi e download file se assenti
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cartella = os.path.join(BASE_DIR, "vector_store")
os.makedirs(cartella, exist_ok=True)

id_faiss = "16zQqSip_8yjAG4UZmJuqUujHcLuckvhi"
id_texts = "1RmKCSe0CtT3zIvhS9dWfqcFMD-56v7Dv"

path_faiss = os.path.join(cartella, "prodotti_index.faiss")
path_texts = os.path.join(cartella, "prodotti_texts.pkl")

scarica_file_gdrive(id_faiss, path_faiss)
scarica_file_gdrive(id_texts, path_texts)

faiss_index = faiss.read_index(path_faiss)

with open(path_texts, "rb") as f:
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
# 5️⃣ Streamlit UI
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

categorie = ["Tutti", "Cancelleria", "Ferramenta", "Cartucce e Stampanti", "Altro"]
categoria = st.selectbox("Seleziona categoria:", categorie)

user_input = st.text_input("Scrivi la tua richiesta:")

if "show_info" not in st.session_state:
    st.session_state.show_info = False

if st.button("Cerca prodotto") and user_input:
    st.info("⏳ Sto cercando i prodotti più adatti...")

    risultati = cerca_prodotti(user_input)
    
    if categoria != "Tutti":
        risultati = [r for r in risultati if categoria.lower() in r.lower()]

    prompt = "Sei un assistente vendita di ferramenta e cancelleria. Rispondi consigliando il prodotto più adatto.\n"
    for ex in few_shot:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    prompt += f"Utente: {user_input}\nAssistente:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    st.subheader("💡 Prodotti consigliati dal vector store:")
    if risultati:
        for i, r in enumerate(risultati, 1):
            st.success(f"{i}. {r}")
    else:
        st.warning("Nessun prodotto trovato per la categoria selezionata.")

    st.subheader("📝 Risposta generata dal modello:")
    st.write(response.choices[0].message.content)

    st.session_state.ultimi_risultati = risultati
    st.session_state.show_info = False

if st.button("Mostra informazioni aggiuntive"):
    st.session_state.show_info = True

if st.session_state.show_info and "ultimi_risultati" in st.session_state:
    st.info("ℹ️ Qui puoi aggiungere dettagli come disponibilità in magazzino, alternative o specifiche tecniche.")
    for i, r in enumerate(st.session_state.ultimi_risultati, 1):
        st.write(f"✅ {r} - Disponibilità: In stock, Sconto attuale: 5%")
