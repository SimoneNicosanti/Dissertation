Nella fase di profiling, bisogna fare il profiling livello per livello per predire il tempo di esecuzione del livello quando mandato in deployment su un certo nodo di rete.

Consideriamo una macchina `c4-standard-4` di google cloud platform con supporto ad OpenVINO.
La misura del tempo è fatta:
- Media su 10 run
- Calcolo con perf_counter_ns
- Provider OpenVINO
- Una run a vuoto (per attenuare gli effetti di cold start)

In questo caso quindi sembra esserci tendenzialmente beneficio: notare che sicuramente la misura del tempo fatta in questo modo presenta dei residui di altre cose fatte in fase di loading del modello, ma possiamo assumere trascurabile.
![[Schermata del 2025-06-18 14-01-39.png]]
I risultati riportati in questo caso sono la somma dei tempi di profilazione su tutti i livelli.

La cosa strana è che se facciamo la misura con i seguenti parametri:
- Media su 10 run
- Calcolo con perf_counter_ns
- Provider CPU (Senza OpenVINO)
- Una run a vuoto (per attenuare gli effetti di cold start)
![[Schermata del 2025-06-18 13-58-21.png]]
Senza il supporto ad OpenVINO, il modello non quantizzato sembra peggiorare le sue prestazioni.

Questa cosa è molto strana considerando che OpenVINO dovrebbe essere ottimizzato per l'esecuzione su Intel. Forse ci sono effetti che non sto considerando, ma non saprei bene dove cercarli...

## Anlisi predittiva dell'ottimizzatore

Stessi parametri di sopra
![[Schermata del 2025-06-18 15-21-54.png]]

Parametri di ottimizzazione:
![[Schermata del 2025-06-18 15-22-31.png]]

Risultati di ottimizzazione:
![[Schermata del 2025-06-18 15-23-05.png]]

Tempo medio di esecuzione su 100 run:
![[Schermata del 2025-06-18 15-24-59.png]]


Parametri di ottimizzazione:
![[Schermata del 2025-06-18 15-26-54.png]]

Risultato ottimizzatore:
![[Schermata del 2025-06-18 15-27-28.png]]

Tempo Medio:
![[Schermata del 2025-06-18 15-30-01.png]]

Sembra quindi che di base il tempo di inferenza sia più basso rispetto a quello predetto dall'ottimizzatore; in aggiunta a questo, sembra che la versione quantizzata non apporti un grandissimo beneficio...

Bisognerebbe quindi capire un attimo come gestire la cosa; è vero che non si può pretendere la massima accuratezza in termini predittivi, ma starci almeno vicini sarebbe una buona cosa.

Nel test che segue abbiamo sempre la stessa configurazione ma con `yolo11l-seg`.
Analisi:
1. La prima inferenza è fatta senza nessun tipo di quantizzazione
2. La seconda quantizzando ma senza mettere pesi e attivazione a quantizzazione simmetrica
3. La terna quantizzando e mettendo pesi e attivazioni simmetriche
![[Schermata del 2025-06-18 16-42-25.png]]
Sembra quindi che l'uso della simmetria (i.e. usare uno zero_point = 0 nella quantizzazione lineare) permette una buona riduzione dei tempi di inferenza.

Stessi risultati si ottengono anche per il modello `yolo11s-seg`: mentre prima i tempi erano maggiori, adesso sono minori.
![[Schermata del 2025-06-18 17-01-59.png]]


> [!NOTE] Title
> Questi tempi non sono casuali: ho eseguito più volte il calcolo e i valori vengono sempre molto vicini tra loro.


> [!Warning] Coerenza
> Ammesso di usare quantizzazione simmetrica, sarà necessario essere coerenti nel suo uso! Questo significa che dovrà essere usata in tutto il "flusso di quantizzazione", quindi sia in fase di profilazione dell'errore, sia in fase di generazione delle componenti, sia in fase di profiling del tempo di esecuzione del modello.

In termini di accuratezza, l'uso di simmetria non dovrebbe creare problemi: nel nostro caso, tutti i livelli che vengono quantizzati sono livelli convoluzionali, che quindi non soffrono di effetti collaterali dovuti alla simmetria (cosa che potrebbe succedere ad esempio in una relu in cui per definizione gli output non possono essere simmetrici). In un contesto più generale potrebbe essere necessario escludere dalla quantizzazione alcuni livelli sensibili che potrebbero causare perdita di accuratezza.


> [!Warning] Analisi del punto di taglio
> Bisogna analizzare bene e con accuratezza il punto migliore in cui è bene tagliare per valutare il tempo dei livelli quantizzati

Se consideriamo la figura seguente, il taglio dovrebbe essere fatto in corrispondenza della parte verdina, proprio per catturare le operazioni di Dequantize e Quantize. Il problema è che così facendo si trascura la parte di QuantizeLinear precedente che serve per la conversione ad int8 dell'input precedente e che comunque è presente. 
![[Schermata del 2025-06-18 16-58-53.png]]

Si potrebbe:
1. Trascurarla e basta, considerando il suo tempo di esecuzione basso (prodotto element-wise completamente parallelizzabile)
2. Considerarlo facendo il taglio a monte
	1. Se lo considero bisogna vedere un attimo poi cosa succede in fase di profiling del tempo di esecuzione (VALUTARE QUESTA COSA!!)

