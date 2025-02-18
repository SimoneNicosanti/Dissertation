## Model Deployment
![[ModelDeployment.svg|Model Deployment]]
Breve descrizione delle interazioni:
1. **Client** richiede il deployment di un modello sul sistema
2. Il **Deployer** contatta la componente di ottimizzazione per ottimizzare il modello richiesto
3. **Optimizer** procede in questo modo
	1. Contatta **Registry** per ottenere la lista dei **server** attualmente attivi
	2. Per ogni **Server** viene contatto un **ServerMonitor** che restituisce informazioni riguardo il server; queste informazioni possono essere del tipo
		1. Banda disponibile e/o metrica di vicinanza agli altri server (più sono "vicino" meno ci metto ad inviare i dati; potrei considerare ad esempio il tempo medio di ping agli altri nodi del sistema)
		2. Memoria RAM residua
		3. Capacità di calcolo
	3. Note queste informazioni, viene fatta l'ottimizzazione del modello: vengono prodotti diversi piani di partizionamento del modello in cui ad ogni parte del modello diviso viene assegnato un server. Qui possiamo tenere in considerazione gli aspetti di:
		1. Quantizzazione
		2. Varianti di modelli (esempio, YOLO)
	4. Le varie parti prodotte vengono salvate nel **ModelPool** assieme a dei metadati di identificazione della versione del modello
	5. Ordina un pull ai diversi **Server** che mandano in servizio quella parte del modello
		1. Il **Client** che avvia l'inferenza, essendo in grado di eseguire il modello è da considerarsi al pari di qualsiasi altro server.


## Inferenza

![[Inference.svg|Inferenza]]
Descrizione delle interazioni:
1. **ClientFrontEnd** è la componente del client che riceve le richieste di inferenza; permette di disaccoppiare il sistema di raccolta delle immagini dall'inferenza sulle stesse
	1. Raccoglie lo stato energetico del dispositivo dal **Monitor** del dispositivo stesso
	2. Invia all'**Optimizer** una richiesta di produzione di piano per l'inferenza
	3. Ricevuto il piano ottimizzato esegue le richieste ai **Server**
		1. Anche qui avremo un'istanza di **Server** in esecuzione nel **Client**
	4. Il risultato viene tornato attraverso un callback da cui poi si salva il risultato, qualora il suo salvataggio sia necessario.
		1. In caso di fallimento, la callback potrebbe avviare di nuovo l'inferenza (posto che possa risalire all'input a cui corrisponde)
2. L'**Optimizer** può ottimizzare il piano. Possiamo considerare tre aspetti.
	1. L'ottimizzazione può essere fatta sulla base di informazioni come:
		1. Latenza massima voluta
		2. Accuratezza minima voluta (o misura correlata)
		3. Consumo energetico massimo / Livello di Consumo (assumendo una divisione in fasce di consumo) (questo aspetto in realtà è anche in parte legato all'accuratezza, quindi uno dei due si potrebbe incorporare nell'altro ??)
	2. La parte di modello in esecuzione sul **Client** dovrebbe essere forzata ad essere la prima e/o l'ultima, in modo che vi sia compressione dei dati nella prima fase e veloce ricezione dell'output nell'ultima
	3. Volendo, si potrebbero produrre piani alternativi: fissata una divisione del modello, se una sotto parte viene mandata in esecuzione su più server, si potrebbero produrre più piani per avere tolleranza ai guasti.


Esempio di Piani:
- Plan_0: esecuzione locale
- Plan_1: l'output del server_0 può essere mandato sia a server_1 che a server_2, per poi continuare. Questo potrebbe essere ad esempio per tolleranza ai guasti: se non risponde 1 lo mando a 2
- Plan_2: l'output di server_0 viene mandato in parte a server_1 e in parte a server_2 per poi ricongiungere il tutto su server_3
![[Plans.svg|Generazione di Piani Diversi|650]]