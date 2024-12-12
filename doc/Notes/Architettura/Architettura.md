
Punto della situazione:
- Device deve/può
	- Eseguire parte della computazione
		- In funzione dello stato energetico
		- Questa parte può essere la prima (feature extraction o una sua sotto parte)
	- Scegliere un modello più o meno accurato
		- In funzione di
			- Condizione della rete
			- Stato della batteria
	- Scegliere una compressione dei dati
		- Per diminuire il tempo di invio (a costo di minore accuratezza nella risposta)
		- La compressione dei dati può essere fatta anche considerando un punto in cui la rete si "assottiglia", cioè i dati che sono elaborati dalla rete diventano meno
- Il modello deve/può essere partizionato per supportare questi aspetti
	- La divisione può essere fatta in funzione della capacità di calcolo/memoria del dispositivo a cui quella parte di modello viene assegnata
	- Un singolo server può anche gestire più parti contemporaneamente e/o più modelli e/o più varianti dello stesso modello (ad esempio quantizzazione)



![[Diagram_1.svg]]
