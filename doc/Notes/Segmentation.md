Risorse per segmentazione:
- https://keras.io/keras_hub/api/base_classes/image_segmenter/

Non sembrano esserci implementazioni di Yolo che supportino anche ImageSegmentation.

## SAM

> [!Bug] Sam e Versioni Varie
> Sembra che l'unnest non funzioni in modo corretto su questa famiglia di reti. Credo ci siano dei problemi con il fatto che i livelli di input hanno dei nomi ripetuti.

![[Schermata del 2024-12-18 17-25-22.png|Esempio di nomi ripetuti]]

Questa cosa è abbastanza strana perché Keras non dovrebbe permettere che ci siano dei nomi di livelli ripetuti all'interno del modello, anche se ci sono dei sotto-modelli.
![[Schermata del 2024-12-18 22-07-06.png|Erreri sugli InputLayer in SAM_Base]]

Il problema che porta la ripetizione del livello è il seguente. Il livello *boxes* viene usato come input layer del modello generale e poi riusato come input layer di uno dei sottomodelli (sequenziali mi sembra di capire); quello che succede quindi è che quando elaboro *boxes* nella ricerca dei nodi successori del grafo, trovo il sub model di cui prendo gli input layer e tra questi input layer c'è proprio *boxes* stesso, portando a ripetizione ricorsiva.
![[Schermata del 2024-12-18 22-30-28.png|Successori Ricorsivi]]


> [!Done] 
> In realtà quel formato mostrato nell'immagine dell'errore credo sia dovuto a Netron: il modello forse è troppo grande per poter essere gestito correttamente.

### Fix
In realtà i due modelli sembrano dare gli stessi risultati


## Modelli Analizzati

### DeepLabV3Plus
La versione pre addestrata in keras_hub non sembra funzionare; sembrano esserci dei problemi su un layer custom non correttamente esportato (o meglio, di cui non si riesce a ricostruire l'istanza partendo dal config).

![[Schermata del 2024-12-22 19-24-15.png|Errore Ricevuto]]

Come si vede dal path del modello si tratta proprio di un livello custom.
![[Schermata del 2024-12-22 19-24-56.png|Punto del codice in cui è l'errore]]

La versione di DeepLabV3Plus in keras_cv invece non sembra dare problemi (probabilmente si basa su livelli standard o che hanno un import/export della configurazione fatto bene); in questo caso il test fatto ha successo.

### SegFormer
In questo caso l'implementazione keras_cv non sembra funzionare: l'unnest del modello funziona senza problemi, quello che non funziona è il suo caricamento!! Anche qui il problema sembra risiedere nella gestione parziale della ricostruzione del layer.

![[Schermata del 2024-12-22 19-28-44.png|Errore Ricevuto]]

![[Schermata del 2024-12-22 19-29-56.png|Punto del Codice e Classe Annessa]]


### Segment Anything Model

In questo caso il modello preaddestrato su keras_hub funziona senza problemi, anche se sarebbe un attimo da capire come usarlo.