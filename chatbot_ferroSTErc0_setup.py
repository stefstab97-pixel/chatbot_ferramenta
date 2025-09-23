import streamlit as st
import pickle
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import gdown
from streamlit_lottie import st_lottie
import requests


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
# Funzione per impostare background da Google Drive
# -----------------------------
def set_background_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{url}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}
        </style>
    """, unsafe_allow_html=True)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.set_page_config(
    page_title="🛠️ Chatbot Ferramenta & Cancelleria",
    page_icon="🛒",
    layout="wide"
)
set_background_from_gdrive("1Y6tHszkZVtKNGwjLpt1346xBsO_0ET5i")


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
# 4️⃣ Few-shot examples di prodotto
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
# 5️⃣ Few-shot consigli e suggerimenti d’uso
# -----------------------------
few_shot_consigli = [
    {"user": "Dimmi come usare un evidenziatore",
     "assistant": "L’evidenziatore Marca X è ideale per sottolineare testi importanti. Consigliamo di usare colori diversi per categorie di concetti."},
    {"user": "Come si usano le etichette adesive?",
     "assistant": "Le etichette adesive si applicano facilmente su superfici lisce. Ottime per organizzare documenti e scatole."},
    {"user": "Indicazioni per usare cartucce per stampante",
     "assistant": "Le cartucce vanno installate seguendo le istruzioni della stampante. Si consiglia di conservare una scorta per evitare interruzioni."},
]


# -----------------------------
# 6️⃣ Setup interfaccia e session state
# -----------------------------
st.title("🛠️ Chatbot Ferramenta & Cancelleria")
st.markdown("""
Benvenuto! Scrivi la tua richiesta e ti suggerirò i prodotti più adatti.
Puoi anche selezionare una **categoria** e una **fascia prezzo** per affinare i risultati.
""")


# Inizializza stati sessione
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_info" not in st.session_state:
    st.session_state.show_info = False
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Filtri
categoria = st.selectbox("Seleziona categoria:", ["Tutti", "Cancelleria", "Ferramenta", "Cartucce e Stampanti", "Altro"], key="categoria")
fascia_prezzo = st.slider("Seleziona fascia prezzo (€):", 0, 1000, (0, 500), key="fascia_prezzo")


# Input testo
st.text_area("Scrivi la tua richiesta:", height=70, key="user_input", placeholder="Scrivi qui... (Premi Invio per inviare)")


# Callback per esempi rapidi aggiorna input testo
def set_user_input(value):
    st.session_state.user_input = value


with st.expander("Esempi rapidi"):
    cols = st.columns(len(few_shot))
    for i, ex in enumerate(few_shot):
        cols[i].button(ex["user"], on_click=set_user_input, args=(ex["user"],))


# Funzioni per gestione chat
def add_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})


def display_chat():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin:10px 0; max-width:70%; font-weight:bold;'>🧑 {chat['message']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin:10px 0; max-width:70%;'>🤖 {chat['message']}</div>",
                unsafe_allow_html=True)


def cerca_e_resetta():
    query = st.session_state.user_input.strip()
    if not query:
        return
    add_message("user", query)
    st.info("⏳ Sto cercando i prodotti più adatti...")

    # Carica animazione Lottie durante il caricamento
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json"
    lottie_json = load_lottieurl(lottie_url)
    lottie_placeholder = st.empty()

    if lottie_json:
        lottie_placeholder.lottie(lottie_json, height=150)

    risultati = cerca_prodotti(query)

    if categoria != "Tutti":
        risultati = [r for r in risultati if categoria.lower() in r.lower()]
    risultati = [r for r in risultati if any(str(p) in r for p in range(fascia_prezzo[0], fascia_prezzo[1] + 1))]

    # Prompt con few shot di prodotto e consigli uso
    prompt = "Sei un assistente vendita di ferramenta e cancelleria. Rispondi consigliando il prodotto più adatto.\n"
    for ex in few_shot:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    for ex in few_shot_consigli:
        prompt += f"Utente: {ex['user']}\nAssistente: {ex['assistant']}\n"
    prompt += f"Utente: {query}\nAssistente:"

    with st.spinner("Elaborazione del modello AI in corso..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

    # Rimuovi animazione Lottie dopo ricezione risposta
    lottie_placeholder.empty()

    response_text = response.choices[0].message.content
    blocchi = [blk.strip() for blk in response_text.split("\n\n") if blk.strip()]

    for idx, blocco in enumerate(blocchi, 1):
        add_message("assistant", f"Proposta {idx}:\n{blocco}")

    st.session_state.show_info = False
    st.session_state.last_feedback = None
    st.session_state.user_input = ""  # reset input


st.button("Cerca prodotto", on_click=cerca_e_resetta)


display_chat()


# Pulsanti feedback risposta assistente
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
    st.write("La risposta ti è stata utile?")
    col1, col2 = st.columns(2)
    if col1.button("Sì"):
        st.session_state.last_feedback = True
        st.success("Grazie per il feedback positivo!")
    if col2.button("No"):
        st.session_state.last_feedback = False
        st.warning("Grazie per il feedback, lavoreremo per migliorare.")


# Pulsante info aggiuntive
if st.button("Mostra informazioni aggiuntive"):
    st.session_state.show_info = True


if st.session_state.show_info:
    st.info("ℹ️ Qui puoi aggiungere dettagli come disponibilità in magazzino, alternative o specifiche tecniche.")
    if "ultimi_risultati" in st.session_state:
        for i, r in enumerate(st.session_state.ultimi_risultati, 1):
            st.write(f"✅ {r} - Disponibilità: In stock, Sconto attuale: 5%")
