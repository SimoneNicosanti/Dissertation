
# Livello di Controllo

![[Architecture_Control.svg]]


## Profiling
Il profiling è fatto di due fasi:
1. Profiling fatto dal manager per:
	1. Collegamenti e costruzione del grafo
	2. Trovare possibili livelli quantizzabili (per FLOPS e/o dimensione out)
	3. Trovare FLOPS del modello
	4. Costruzione del modello di regressione (diventa un attributo del grafo)
2. Profiling fatto dal server
	1. Trovare il tempo di esecuzione di ciascun livello del modello
		1. Senza Quantizzazione
		2. CON QUANTIZZAZIONE (SOLO PER I PAPABILI)

In alternativa potrei fare il profiling di tutti i livelli con e senza quantizzazione e poi in fase di ottimizzazione considerare solo quelli con attributo *quantizable*: forse questa soluzione è migliore perché in questo modo riduco la sequenzialità del profiling. Posso magari trascurare livelli per cui non ha senso fare la profilazione, ad esempio i livelli di "Concat" o "Divide", ma prendere solo livelli 

![[Diagram.svg]]

Per quanto riguarda la divisione del modello, mi conviene creare una componente ModelDivider responsabile della sola divisione; le tre componenti:
- ModelPool
- ModelDivider
- ModelProfiler
Vengono poi mandate in esecuzione tutte sullo stesso server ModelManager che è quello che gestisce poi il tutto: queste componenti per poter funzionare hanno tutte bisogno del modello.
Essendo tutte in esecuzione sullo stesso modello, anche se stiamo usando gRPC, comunque l'overhead di trasmissione non dovrebbe essere eccessivo.
# Livello di Ottimizzazione

![[Architecture_Optimization.svg]]

# Livello di Inferenza

![[Architecture_Inference.svg]]