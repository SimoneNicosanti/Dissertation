## Modello di Rumore
I problemi principali dell'uso della quantizzazione per livelli singoli è il seguente:
1. Non sappiamo come le quantizzazioni tra livelli interagiscono tra di loro
2. Ci sono molte possibili combinazioni di quantizzazione: se ogni livello della rete può essere quantizzato e non quantizzato, allora ci sono $2^{|V_M|}$ possibilità, troppo per indagarle tutte 
3. Il tempo per la valutazione di una certa combinazione di quantizzazione non è poco: per ogni combinazione bisogna, calibrare, quantizzare e poi valutare l'errore

Quello che si potrebbe fare è considerare un sottoinsieme di livelli della rete che siano significativi per la quantizzazione:
- Livelli che hanno un carico alto (alto numero di flops)
- Livelli che hanno un output grande in dimensione (ne beneficia il tempo di trasmissione)
E quantizzare SOLO questo sottoinsieme, imponendo che gli altri non lo siano.

Sia quindi $V_{Q} \subseteq V_M$ dove $V_Q = \{ i \in V_n : i.quantizable = True \}$ il sottoinsieme di possibili livelli quantizzabili.

Comunque anche con pochi livelli (ad esempio tra i 12 e 15), il numero di possibilità è troppo alto per poter essere precisi. Si può introdurre un modello di regressione tale che:
$$
e : \{0,1\}^{|V_Q|} \rightarrow \mathbb{R}^{\ge 0}
$$

Di base serve un modello di regressione che sia esprimibile con vincoli lineari: a questo scopo si potrebbe usare un albero o una semplice regressione lineare.

Alcune prove fanno vedere che (almeno per Yolo11n-seg) una regressione polinomiale funziona bene.

Sia 
Supponiamo di usare una regressione polinomiale. Sia $\hat{q} = (q_1, q_2, ..., q_{|V_Q|})$ il vettore delle variabili $q_i$ solo dei livelli quantizzabili: questa variabile ha valore 1 sse il livello in questione è quantizzato.

Allora possiamo scrivere la funzione di errore come:
$$
e(\hat{q}) = w^T \cdot \hat{q} + c
$$
Per fare in modo che $e(0) = 0$, facciamo le seguenti modifiche:
- $w^T \rightarrow [w^T, c]$
- $\hat{q} \rightarrow [\hat{q}, p]$ 

Dove $p$ è una variabile aggiuntiva tale che $p = 1$ sse $\exists i : q_i = 1$; da cui i vincoli diventano:
- $p \ge q_i$   $\forall i \in V_Q$
- $p \le \sum_{i \in V_Q} q_i$

Da cui quindi otteniamo:
$$
e(\hat{q}) = w^T \cdot \hat{q}
$$
In questo modo la funzione di errore vale zero se nessun livello è quantizzato. Possiamo quindi aggiungere il seguente vincolo alla formulazione:

$$
e(\hat q) \le e_{max}
$$
Dove $e_{max}$ è il massimo errore che si è disposti ad accettare.

La regressione (e quindi i pesi $w^T$ e $c$) si dovrebbe calcolare in fase di deployment del modello: a seconda di:
- Numero di istanze di calibrazione 
- Numero di livelli quantizzabili
Il tempo per trovare questa parametrizzazione cambia


Altri vincoli da aggiungere sono:
$$
q_i = 0 \hspace{0.5cm} \forall i \notin V_Q
$$


Facendo alcune prove si vede che un modello quadratico funziona molto meglio (quasi al pari di un albero) probabilmente perché riesce a tenere in conto le interazioni. Da notare che questo è valido in QUESTO contesto, potrebbe non esserlo in altri in cui magari le interazioni sono molto più sparse.

Nel caso di una regressione quadratica, la cosa diventa leggermente più complicata, ma non troppo (anche se non scala). In questo caso si potrebbe usare la seguente strategia.

In questo caso dobbiamo tenere in conto i prodotti delle variabili. Notiamo subito che essendo le variabili binarie, vale che $(q_i)^2 = q_i$   $\forall i \in V_Q$, quindi le interazioni di un termine con se stesso si possono trascurare senza problemi. Allo stesso modo abbiamo che $q_i \cdot q_j = q_j \cdot q_i$  $\forall i \neq j$ , quindi possiamo considerare metà dei prodotti, per un totale di:
$$
\binom{|V_Q|}{2}
$$
variabili prodotto da considerare. Sia data $q_{ij} = q_i \cdot q_j$ la variabile prodotto; questa variabile è soggetta a vincoli:
1. $q_{ij} \le q_i$
2. $q_{ij} \le q_j$
3. $q_{ij} \ge q_i + q_j - 1$

In totale quindi il regressore è esprimibile come:
$$e(\hat q) = w^T \cdot \hat q = \sum_{i \in V_Q} w_{ii} \cdot q_i + \sum_{i < j} w_{ij} \cdot q_{ij}
$$
Ed è quindi ancora una funzione lineare. Notiamo che se in fase di addestramento le interazioni combinate sono poco significative (i.e. $w_{ij} = 0$) allora non ci sarà bisogno di definire la variabile prodotto e il numero di variabili/vincoli diminuisce.

Con regressioni di livello superiore la cosa è estendibile facilemente (fino ad un certo punto per non far esplodere il problema).