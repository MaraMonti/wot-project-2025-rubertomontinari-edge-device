#wot-project-2025-rubertomontinari-edge-device

**⚡ **Previsione della domanda energetica della casa intelligente****


**A) Descrizione Generale del Progetto**

Il progetto Smart Home Energy Demand Prediction mira a sviluppare un sistema avanzato per la previsione accurata del consumo energetico giornaliero di un'abitazione.

La problematica affrontata è la necessità di ottimizzare i consumi energetici e mitigare gli sprechi, un fattore cruciale sia a livello economico che ambientale.

La soluzione proposta si basa sull'applicazione di tecniche di Machine Learning (ML) avanzate, in grado di identificare pattern complessi nel consumo. Il modello sfrutta:

Un dataset storico fornito.

Dati in tempo reale acquisiti tramite API (consumo, temperatura, umidità).

L'integrazione di dataset esterni (es. dati meteorologici da Tomorrow.io) per migliorare l'accuratezza e la generalizzazione.

**B) Architettura del Sistema**

Il sistema è strutturato come un'applicazione web interattiva basata su Streamlit, che funge da interfaccia per un flusso di dati e calcoli predittivi.

Architettura:

Fonti dei dati: Raccolta dati da diverse sorgenti:

Dataset iniziale: Dataset storico fornito per l'abitazione di Lecce.

API Home Assistant: API Rest per l'acquisizione di dati istantanei di consumo, temperatura e umidità in tempo reale.

Tomorrow.io API: API per dati meteorologici (previsioni e storici).

Modulo Elaborazione Dati: Modulo di pre-elaborazione (data_preprocessing.py) responsabile della pulizia, unificazione (fusione di ~380.000 record da fonti multiple come Kaggle e UMass) e Feature Engineering (creazione di variabili temporali come hour, day_of_week, is_weekend).

Streamlit Application (Frontend): Interfaccia utente che carica il modello ML, visualizza i dati storici, mostra i dati in tempo reale e genera le previsioni.


<img width="877" height="514" alt="Image" src="https://github.com/user-attachments/assets/a774f1ff-40f4-4569-bb08-25405fbea028" />

---

**C) Repository e Componenti**
Il progetto è suddiviso in diversi repository, ciascuno per un componente specifico. Di seguito l'elenco dei repository con i rispettivi link:
| Componente | Ruolo | Repository Link |
| :--- | :--- | :--- |
| **Edge Device** | Rilevazione locale (simulata o reale) dei dati. | **https://github.com/MaraMonti/wot-project-2025-rubertomontinari-edge-device** |
| **Gateway / ML Service** | Contiene il Modello ML e l'App Streamlit (questo repository). | **https://github.com/MaraMonti/wot-project-2025-rubertomontinari-gateway-ml** |
| **Front-end / App** | (Se separato) Interfaccia utente finale. | **[LINK REPO FRONT-END PLACEHOLDER]** |


*(**NOTA:** I link sopra sono placeholder. Sostituirli con gli URL dei repository effettivi.)*

---

## D) Dettaglio Componente: Dispositivo Edge

Questo repository contiene il codice sorgente (principalmente `realtime_data.py`) che gestisce la componente Edge, responsabile dell'acquisizione dei dati in tempo reale e del loro invio al Servizio Gateway/ML per la previsione.

### Funzione Principale

La funzione principale dell'Edge Device è:

1.  **Generazione/Acquisizione Dati:** Il dispositivo simula l'acquisizione di **temperatura**, **umidità** e **consumo energetico** dall'abitazione.
2.  **Preparazione del Payload:** I dati vengono formattati in un **payload JSON** standardizzato.
3.  **Comunicazione:** Il payload viene inviato al Gateway/ML service.

### Dettagli Tecnici di Invio

* **Protocollo:** L'invio avviene tramite richiesta **HTTP POST**.
* **Frequenza:** I dati vengono inviati ogni ora .
* **Destinazione:** Il payload viene indirizzato all'endpoint `/api/data_ingestion` del servizio Gateway/ML per l'elaborazione.
* **File:** Il file principale responsabile dell'invio è `realtime_data.py`.
