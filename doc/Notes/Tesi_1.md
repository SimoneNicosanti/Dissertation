
## Analisi del Modello
Analizzando l'attributo _config_ del modello (che restituisce un oggetto strutturato come un json che descrive il modello) sono riuscito a capire il problema relativo a quei livelli di Add e Multiply che, come mostrato in Netron, prendevano in input da un solo livello, cosa che risultava strana trattandosi di livelli aggreganti; il problema è che questi livelli in realtà prendono in input l'output di un livello precedente per poi eseguire un'operazione element-wise (appunto una somma o un prodotto) con una costante (nello specifico, per la somma la costante è sempre 3.0 mentre per la moltiplicazione è sempre 0.16 con 6 periodico). Quei livelli che venivano mostrati in netron in sostanza non sono affatto dei livelli (cioè istanziati in fase di costruzione del modello come dei keras.Layer), ma sono il risultato di operazioni del seguente tipo:  
```
x1 = prevLayer(prevPrevOutput)  
x2 = x1 + 3  
nextLayer(x2)  
```

Quando viene fatta l'operazione di somma, questa non viene mappata su un oggetto di tipo layer all'atto della compilazione del modello, ma su un oggetto keras diverso, cioè un'istanza di una classe del modulo _keras.src.ops.numpy_ che mappano delle operazioni element-wise in keras.

Premettendo che non mi è molto chiaro il motivo per cui siano necessarie operazioni di questo tipo, in quanto riterrei più sensato imparare dei pesi del modello che fanno direttamente in modo, prendendo per riferimento l'esempio precedente, che x1 sia direttamente uguale al suo valore sommato 3 piuttosto che fare questo. Comunque, facendo il parsing del _config_, sono riuscito ad estrarre la classe di appartenenza di questi oggetti e farne il wrap all'interno di oggetti che si comportassero in modo simile a dei livelli, in modo da poterli distribuire al pari di livelli "classici".

> [!Warning] Validità e generalizzazione della soluzione
> A questo punto il dubbio che mi sorge è relativo alla validità di una soluzione di questo tipo in un contesto più generale: potrebbero esserci altri oggetti con cui keras fa cose simili di cui potrei non venire a sapere l'esistenza e che potrebbero dare problemi simili. Ad ogni modo, ho provato a distribuire anche altri modelli (come diverse versioni di ResNet) e sembrano funzionare con la distribuzione fatta in questo modo, però il dubbio rimane.


## Analisi di protocolli di Serializzazione usati da RPyC
Formato usato *Brine*.

> [!Quote] Brine
> _Brine_ is a simple, fast and secure object serializer for **immutable** objects. The following types are supported: `int`, `bool`, `str`, `float`, `unicode`, `bytes`, `slice`, `complex`, `tuple` (of simple types), `frozenset` (of simple types) as well as the following singletons: `None`, `NotImplemented`, and `Ellipsis`.
> https://rpyc.readthedocs.io/en/latest/api.html#serialization

Potrebbe non essere adatto al trasferimento di dati complessi perché c'è il doppio passaggio, uno per serializzare e altro per convertire in Brine.

## Implementazione distribuzione con gRPC
Controllare l'articolo per vedere le differenze nell'uso del JSON e di protocol buffer
https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62

> [!Warning] Massima dimensione messaggio gRPC
> Problema nell'uso di gRPC: la massima dimensione del messaggio è 4 MB, quindi o si aumenta il limite oppure si passa ad un invio in Stream in caso di batch o input particolarmente grandi

> [!Warning] Problema con il numero di thread gestibili
> In gRPC si deve specificare il numero di thread che possono eseguire un certo servizio. Se sono meno del numero di layer gestiti da un'istanza di servizio si rischia il blocco. Questo aspetto deve essere gestito in qualche modo.

Per risolvere il secondo problema si potrebbero usare delle chiamate non bloccanti, oppure si fa tornare l'output parziale al front-end che poi lo invia al prossimo layer (troppo passaggio dati e troppa latenza forse...). Altrimenti si potrebbe fare implementando delle call asincrone implementazione di call asincrone

Contesto di esecuzione: 
* Se il livello successivo è gestito dallo stesso server viene fatta direttamente la chiamata locale
* Non ci sono più richieste e quindi non ci sono nemmeno dei meccanismi di sincronizzazione sulle strutture dati condivise del server (anche se comunque vengono accedute in lettura)

Nel complesso i tempi di esecuzione sembrano migliori rispetto ad un'esecuzione con RPyC: tempi di esecuzione per 50 esecuzioni.

|         | Locale              | gRPC                | RPyC               |
| ------- | ------------------- | ------------------- | ------------------ |
| Mean    | 0.5486299920082093  | 1.4203560161590576  | 2.472412314414978  |
| Std Dev | 0.04414955138328341 | 0.12214391166816921 | 0.6125166100820715 |
Rapporti

| gRPC / Locale | RPyC / Locale |
| ------------- | ------------- |
| 2.6296        | 4.5065        |
