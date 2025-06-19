Rivedere/Testare bene:
- Profiling per livello
	- Potrebbe non essere molto veritiero!
		- Quando faccio partire l'inferenza vengono fatte delle operazioni di ottimizzazione per cui i livelli poi potrebbero cambiare (esempio di fusioni)
		- Questo potrebbe non essere un problema di base:
			- Semplicemente il modello non cattura questo aspetto del runtime e quindi per forza di cose la soluzione che viene trovata è sub-ottima
		- La soluzione per avere una cosa più veritiera potrebbe essere quella di far fare al grafo l'ottimizzazione in fase preliminare e salvare il grafo.
			- In questo modo si lavora sul modello già ottimizzato
			- Il problema è che non tutti i livelli sono supportati dal tool che fa l'analisi dei flops
				- L'analisi dei flops non serve più a molto di base, mi serve solo per trovare i livelli più pesanti in termini computazionali e valutarli per la quantizzazione
				- La soluzione a questo potrebbe essere che invece di valutare per calcolo, valutiamo per dimensione dell'output, considerando la quantizzazione più come meccanismo di compressione che come meccanismo di velocizzazione del calcolo
					- Questo ha senso anche considerando il fatto che stiamo quantizzando un sottoinsieme dei livelli e che quindi il guadagno potrebbe essere scarso in termini di tempo di calcolo (soprattutto in dei runtime ottimizzati)
			- Questo permette anche di ridurre il numero di livelli del modello (vengono fatte delle fusioni in operatori ottimizzati), quindi permette una buona semplificazione del problema (esempio il modello large passa da 620 nodi circa a 450 circa, mentre il modello x passa a 470).

- Capire bene un attimo come fare per la quantizzazione:
	- Se considerare i livelli a maggiore carico di computazione o meno
	- Se consideriamo i livelli a maggiore carico di trasmissione in realtà la cosa non si attenua molto...
		- Ci sono i blocchi Conv --> Sigmoid / Mul che hanno tutti le stesse dimensioni, quindi considerarli così porterebbe a ridurre 


- Modellazione della trasmissione alla stessa macchina
	- Ci sono due casi:
		- Livelli in stessa componente
			- In questo caso il tempo di trasmissione dovrebbe essere nulla
		- Livelli su stessa macchina ma in componenti diverse
			- In questo caso il tempo di trasmissione va con la velocità dell'interfaccia di loopback

Per la quantizzazione, provare a vedere se l'uso di quantizzazione QOperator migliora i tempi di inferenza: forse va leggermente meglio, ma non è detto perché c'è comunque la latenza di conversione aggiuntiva. COMUNQUE VALE LA PENA DI PROVARE QUESTA COSA!! FORSE HO UNO SCONTO MAGGIORE DEL COSTO (bisogna individuare i nomi dei livelli da tagliare però).