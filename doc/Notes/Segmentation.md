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


## Altri Modelli
L'uso degli altri modelli non sembra dare problemi con l'Unnest e i risultati ottenuti sono gli stessi sia con il modello originale che non.