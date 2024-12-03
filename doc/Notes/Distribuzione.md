## Protocollo di Serializzazione usato da RPyC

Formato usato _Brine_.

> [!Note] Brine
> _Brine_ is a simple, fast and secure object serializer for **immutable** objects. The following types are supported: `int`, `bool`, `str`, `float`, `unicode`, `bytes`, `slice`, `complex`, `tuple` (of simple types), `frozenset` (of simple types) as well as the following singletons: `None`, `NotImplemented`, and `Ellipsis`.
> https://rpyc.readthedocs.io/en/latest/api.html#serialization

Potrebbe non essere adatto al trasferimento di dati complessi perché c'è il doppio passaggio, uno per serializzare e altro per convertire in Brine.

## Implementazione distribuzione con gRPC

### Distribuzione per Livello Singolo

Controllare l'articolo per vedere le differenze nell'uso del JSON e di protocol buffer
https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62

> [!Warning] Massima dimensione messaggio gRPC
> Problema nell'uso di gRPC: la massima dimensione del messaggio è 4 MB, quindi o si aumenta il limite oppure si passa ad un invio in Stream in caso di batch o input particolarmente grandi

> [!Warning] Problema con il numero di thread gestibili
> In gRPC si deve specificare il numero di thread che possono eseguire un certo servizio. Se sono meno del numero di layer gestiti da un'istanza di servizio si rischia il blocco. Questo aspetto deve essere gestito in qualche modo.

Per risolvere il secondo problema si potrebbero usare delle chiamate non bloccanti, oppure si fa tornare l'output parziale al front-end che poi lo invia al prossimo layer (troppo passaggio dati e troppa latenza forse...). Altrimenti si potrebbe fare implementando delle call asincrone implementazione di call asincrone

Contesto di esecuzione:

- Se il livello successivo è gestito dallo stesso server viene fatta direttamente la chiamata locale
- Non ci sono più richieste e quindi non ci sono nemmeno dei meccanismi di sincronizzazione sulle strutture dati condivise del server (anche se comunque vengono accedute in lettura)

Nel complesso i tempi di esecuzione sembrano migliori rispetto ad un'esecuzione con RPyC.

| Dati su 50 Run | Niente  | gRPC    | RPyC    |
| -------------- | ------- | ------- | ------- |
| Mean           | 0.54863 | 1.42036 | 2.47241 |
| Std Dev        | 0.04415 | 0.12214 | 0.61252 |
| x / Locale     | 1       | 2.58892 | 4.5065  |

### Distribuzione per Sotto Modelli usando gRPC

Implementazione del servizio con gRPC e per sotto modelli (invece che per singolo livello).

Riuscito abbastanza tranquillamente. Aspetti significativi dell'implementazione:

- Interazione Registry-Server
  - Il server si registra sul registry; il registry risponde con un indice che rappresenta la porzione di modello che quel server deve prendere in carico
  - Il server analizza l'input del suo sotto modello e invia al registry quali sono i livelli di input del suo sotto modello
  - Il registry registra questi livelli associandoli alla coppia (indirizzo IP - Porta) del server in questione
- Interazione Client-Server
  - Il Client genera il suo input
  - Chiede al registry dove si trova l'input layer
  - Invia l'input al modello
  - Riceve dizionario di input con i diversi output del modello
- Elaborazione del Server
  - Il server riceve un messaggio di input
    - Per il requestId ricevuto si salva quell'input in un dizionario
    - Se ho ricevuto tutti gli input per il mio sotto modello per una specifica richiesta
      - Elaboro l'input e ricevo un dizionario di output
      - Per ogni livello di output
        - Chiedo al registry dove si trova il livello di input corrispondente
        - Invio l'output
    - Se mi manca qualche input
      - Ritorno un output vuoto

Contesto di esecuzione:

- 100 Run
- 5 Server, ognuno con una porzione del modello diversa
- Unico client (senza richieste concorrenti)
- Niente GPU
- Flusso completamente delegato ai server (le risposte non tornano indietro al client ma sono i server che si passano il risultato intermedio della computazione)

| Dati su 100 Run | Niente  | gRPC    |
| --------------- | ------- | ------- |
| Mean            | 0.52443 | 0.60116 |
| Std Dev         | 0.02925 | 0.02863 |
| x / Locale      | 1       | 1.14631 |

L'impatto della distribuzione è nel complesso molto minore rispetto all'implementazione precedente: nel caso precedente l'input veniva passato per ogni layer e quindi lo stack di chiamate si accresceva appesantendo l'esecuzione. In questo caso ad una chiamata corrisponde l'esecuzione dell'intero sotto modello, quindi

> [!Bug] Ultimo Layer
> Bisogna fare attenzione a come identificare l'ultimo layer. Non posso identificarlo come quel layer che non ha successori, perché il registry potrebbe rispondermi picche anche per nodi gestori dei successivi che sono caduti nel frattempo.
>
> Si potrebbe gestirlo con un nome fisso (come nel caso dell'input-layer), ma perderei flessibilità per modelli a più output.
>
> Lo potrei aggiungere in dei metadati del modello che viene preso in carico da un certo server.

> [!Warning] Da aggiungere
> Aggiungere il fatto che il registry gestisce una delle liste per quanto riguarda gli input attesi. Dato un input di un sotto modello possono esserci più sotto modelli che lo aspettano.

### Distribuzione per Sotto Modelli con TF Serving

Se tutto ha senso come dovrebbe, si potrebbe fare anche la divisione e far gestire ogni sotto modello a dei TF Server.
Aspetti da considerare:

- Distribuzione dell'input
  - Si potrebbe creare una componente di ricezione dell'input che aspetta di ricevere l'input completo del sotto modello per poi chiamare TF Serving che lo esegue
- L'output del sotto modello non viene inviato subito al sotto modello successivo, ma torna al chiamante
  - A meno di non gestire in queste componenti di arrivo anche l'aspetto di invio dell'output ai sotto modelli successivi

Comando per lanciare il TF Serving

```shell
tensorflow_model_serving --port={x gRPC} --rest_api_port={} --model_name={} --model_base_path={}

```
