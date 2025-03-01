L'ottimizzazione viene modellata come problema di assegnamento di un grafo logico (il modello) su un grafo fisico (la rete).

Sia $G_M = (V_M, E_M)$ il grafo del modello e sia $G_N = (V_N, E_N)$ il grafo della rete. Quello che vogliamo è mappare ogni nodo del modello su un nodo della rete e mappare ogni arco del modello su un arco della rete. Nello specifico siano:
- $G_M$ un grafo diretto aciclico
- $G_N$ un grafo diretto. Supponiamo che all'interno dell'insieme $E_N$ siano presenti anche archi di tipo $(v, v)$ $\forall v \in V_N$ per modellare il passaggio di dati da un nodo a se stesso.

Sia dato un nodo $v \in V_M$. Definiamo per questo livello una tupla $\phi_v$ di caratteristiche del nodo; in particolare sia $\phi_v = (flops\_livello)$ .

Sia dato un nodo $v \in V_N$. Definiamo per questo livello una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia $\psi_v = (flops\_al\_secondo, consumo\_energetico\_calcolo, consumo\_energetico\_trasmissione)$.

Sia dato un arco $e \in E_M$. Definiamo per questo arco una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia $\eta_e = (size\_dati\_arco)$.

Sia dato un arco $e \in E_N$. Definiamo per questo arco una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia $\epsilon_e = (banda\_arco)$.

Definiamo le variabili:
- $x_{ik} \in \{0, 1\}$ tc   $x_{ik} = 1$ sse $i \in V_M$ assegnato a $k \in V_N$
- $y_{ab} \in \{0, 1\}$ tc  $y_{ab} = 1$ sse $a \in E_M$ assegnato a $b \in E_N$

Notare che per come è modellata la rete di server, i server potrebbero non essere connessi in una mesh: si potrebbe anche pensare a delle connessioni per cui, pensando ad una divisioni in livelli del continuum (edge, fog, cloud), ogni livello è connesso solo a nodi di livelli superiori e a nodi dello stesso livello, ovvero la computazione non può "tornare indietro" (a meno di non voler spostare la parte finale della computazione sul nodo iniziale per non dover ritrasmettere il risultato).


> [!TODO] 
> Controllare normalizzazione per parte di energia e parte di latenza.
> In teoria devo escludere le variabili dalla normalizzazione, perché altrimenti il problema è circolare: dovrei prendere il massimo come se le assegnazioni fossero fatte sembre.


# Modello di Latenza

## Latenza di Calcolo
La latenza di calcolo di un livello su un nodo della rete la possiamo calcolare come segue; dato $k \in V_N$ abbiamo

$$t_k^{c} = \sum_{i \in V_M} \frac{\phi_{i0}}{\psi_{k0}} x_{ik}$$

Normalizzando otteniamo:
$$\hat{t}_k^{c} = \frac{t_k^c}{\max_{h \in V_N} t_h^c} $$

Da cui, la latenza complessiva dovuta al calcolo su tutti i server è data da:
$$T^{c} = \sum_{k \in V_N} \hat{t}_{k}^{c}$$


## Latenza di Trasmissione
La latenza di trasmissione dipende invece dalla somma delle latenze di trasmissione in funzione delle bande dei link fisici a cui i link logici sono collegati. 

$$t_{k}^{x} = \sum_{a = (i, j) \in E_M} \hspace{0.1cm} \sum_{b \in E_N \wedge k == b[0]} \frac{\eta_{a0}}{\epsilon_{e0}}y_{ab}$$

Normalizzando otteniamo:
$$\hat{t}_k^{x} = \frac{t_k^x}{\max_{h \in V_N} t_h^x} $$

Da cui la latenza totale dovuta alla trasmissione è data da:
$$T^x = \sum_{k \in V_N} \hat{t}_{k}^{x}$$

## Latenza totale
La latenza totale è quindi data da
$$T = T^c + T^x$$
# Modello di Energia

## Energia di Calcolo
L'energia data dal calcolo per $k \in V_N$ è data da:
$$E_k^c = t_k^c * \psi_{k1}$$

Normalizzando:
$$\hat{E}_k^c = \frac{E_k^c}{\max_{h \in V_N} E_h^c}$$

Da cui l'energia complessiva data dal calcolo è data da
$$E^c = \sum_{k \in V_N} \hat{E}_k^c$$


## Energia di Trasmissione
L'energia data dalla trasmissione per $k \in V_N$ è data da:
$$E_k^x = t_k^x * \psi_{k2}$$
Normalizzando:
$$\hat{E}_k^x = \frac{E_k^x}{\max_{h \in V_N} E_h^x}$$

Da cui l'energia complessiva data dal calcolo è data da
$$E^x = \sum_{k \in V_N} \hat{E}_k^x$$
## Energia Totale
$$E = E^x + E^c$$


# Modello di Memoria


# Modello di Penalità di Quantizzazione


# Problema
Indichiamo con:
- $V_I \subset V_M$ il sottoinsieme di nodi di input; 
- $V_O \subset V_M$ il sottoinsieme di nodi di output

Siano:
- $\alpha_c$ e $\alpha_x$ i pesi della latenza di calcolo e di trasmissione
- $\beta_c$ e $\beta_x$ i pesi dell'energia di calcolo e di trasmissione
Tali che:
$$\sum_{i \in \{x,c\}} \alpha_i + \sum_{i \in \{x,c\}} \beta_i = 1$$

In definitiva il problema diventa il seguente:
$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \alpha_c*T^c + \alpha_x*T^x + \beta_c*E^c + \beta_x*E^x \\
\\
Vincoli:\\
\sum_{k \in V_N} x_{ik} = 1 \hspace{1cm}  \forall i \in V_M \\
\sum_{b \in E_N} y_{ab} = 1 \hspace{1cm} \forall a \in E_M \\
x_{ik} = \sum_{h \in V_N} y_{(i,j)(k,h)} \hspace{1cm} \forall a=(i,j) \in E_M, \forall k \in V_N \\
x_{jh} = \sum_{k \in V_N} y_{(i,j)(k,h)} \hspace{1cm} \forall a=(i,j) \in E_M, \forall h \in V_N \\
x_{i0} = 1 \hspace{1cm} \forall i \in V_I \\ 
x_{j0} = 1 \hspace{1cm} \forall j \in V_O \\
\\
x_{ik} \in \{0, 1\} \hspace{1cm} \forall i \in V_M, \forall k \in V_N \\
y_{ab} \in \{0, 1\} \hspace{1cm} \forall a \in E_M, \forall b \in E_N \\

\end{matrix}

$$
Analisi vincoli:
1. Il primo vincolo serve ad imporre che ogni nodo del modello sia mappato su esattamente un server
2. Il secondo vincolo serve ad imporre che ogni arco del modello sia mappato su esattamente un arco della rete di server
3. Il terzo e il quarto vincolo sono vincoli di flusso
	1. Il terzo dice che se $i \in V_M$ è assegnato a $k \in V_N$, allora i dati prodotti da $i$ sono inviati ai server che hanno i nodi successori di $i$
	2. Il quarto vincolo è simile al terzo ma per la ricezione
4. Il quinto vincolo impone che almeno i livelli di input si trovino sul server 0 che fa partire l'inferenza (per assunzione)
5. Il sesto vincolo impone che i livelli di output si trovino sul server 0 che fa partire l'inferenza (per assunzione)


# Test dell'Implementazione

^91b978

Test del modello solo con latenza, escludendo l'energia

Parametri di test:
- Due server
	- Server_0 = (flops = 10)
	- Server_1 = (flops = 11)
- Link tra i server
	- Banda (bidirezionale) = 1000 (B/s)
- Modello
	- ResNet50, tagliato come in figura
	- Flops dei nodi tutti impostati a 1000
	- Size input assunta (1, 3, 448, 448)
- Si è aggiunto come vincolo aggiuntivo che anche il primo livello convoluzionale si trovi su server 0 (vedere dimensioni)

La normalizzazione dell'obiettivo è fatta min-max.

![[Schermata del 2025-02-22 09-43-29.png|Sotto modello di Test|200]]

Nota le dimensioni di out della tabella sono scorrette, fare riferimento alle proporzioni che stanno sotto la tabella.
![[Schermata del 2025-02-22 09-45-41.png|Dimensioni e Tempi di trasmissione]]
Come si vede la dimensioni dei dati dal primo nodo al secondo è molto piccola, quindi verrebbe trasferita immediatamente, spostando tutta la computazione sul server 1: per questo si vincola che anche quel nodo stia sul primo server.


Divisione ottenuta dall'ottimizzatore:

| ![[Schermata del 2025-02-22 09-47-59.png\|Server 0]] |
| ---------------------------------------------------- |
| ![[Schermata del 2025-02-22 09-48-33.png\|Server 1]] |
Come si vede dal risultato ottenuto la computazione rimane sul server 0 fino all'esecuzione della MaxPool, a seguito della quale si ottiene una dimensione dei dati per cui risulta conveniente spostare il calcolo sul server_1 più veloce piuttosto che rimanere sul server_0 più lento.

## Astrazione del Modello
Sia il modello sia la rete su cui questo viene mandato in deployment vengono modellati come dei grafi.

All'interno del grafo del modello sono aggiunti due nodi fittizi, ovvero ModelInputNode e ModelOutputNode, che servono a semplificare la gestione della posizione dell'input e dell'output (input ed output si trovano sul server che avvia l'inferenza per default).

Allo stesso modo definiamo degli archi di input e di output sul modello che servono a trasportare l'input e l'output tra i vari sottomodelli (o da/verso il modello completo).



# Ottimizzazione 2.0

L'ottimizzazione viene modellata come problema di assegnamento di grafi logici (il modello e le sue varianti) su un grafo fisico (la rete di server).

Sia dato un task $T$ e sia $A$, l'insieme di varianti di un modello $M$ per la risoluzione del task; sia quindi $A = \{G_M^1, G_M^2, ..., G_M^{|A|}\}$ l'insieme delle varianti. Sia definito $A_{idx} = \{x : G_M^{x} \in A\}$ l'insieme degli indici delle varianti nell'insieme $A$.

Dato un $a \in A_{idx}$ definiamo $G_M^a = (V_M^a, E_M^a, \delta^a)$ il grafo del modello; nello specifico:
- La coppia $(V_M^a, E_M^a)$ rappresenta un DAG
- Il valore $\delta^a$ rappresenta l'accuratezza del modello $a$-esimo

Dato un $a \in A_{idx}$, definiamo $\lambda^a$ come il numero di richieste fatte al sistema nell'unità di tempo per la variante $a$-esima.

 Sia $G_N = (V_N, E_N)$ il grafo della rete. $G_N$ è un grafo diretto tale per cui supponiamo che all'interno dell'insieme $E_N$ siano presenti anche archi di tipo $(v, v)$ $\forall v \in V_N$ per modellare il passaggio di dati da un nodo a se stesso.

Quello che vogliamo è: $\forall a \in A_{idx}$ mappare ogni nodo e ogni arco di $G_M^a$ rispettivamente su un nodo e su un arco della rete $G_N$.

> [!NOTE] Connessioni di Rete
> Notare che per come è modellata la rete di server, i server potrebbero non essere connessi in una mesh: si potrebbe anche pensare a delle connessioni per cui, pensando ad una divisioni in livelli del continuum (edge, fog, cloud), ogni livello è connesso solo a nodi di livelli superiori e a nodi dello stesso livello, ovvero la computazione non può "tornare indietro" (a meno di non voler spostare la parte finale della computazione sul nodo iniziale per non dover ritrasmettere il risultato).

Sia dato un $G_M^a \in A$. Definiamo:
- Dato un nodo $v \in V_M^a$, una tupla $\phi_v$ di caratteristiche del nodo del modello; in particolare sia:
	- $\phi_{v0}$ i flops del livello
	- $\phi_{v1}$ la memoria dovuta ai pesi del livello
	- $\phi_{v2}$ la memoria dovuta all'output
- Dato un arco $e \in E_M^a$, una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia:
	- $\eta_{e0}$ la size dei dati che caratterizzano l'arco

Consideriamo $G_N$. Definiamo:
- Dato un nodo $v \in V_N$, una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia:
	- $\psi_{v0}$ i flops al secondo calcolabili
	- $\psi_{v1}$ il consumo energetico dato dal calcolo
	- $\psi_{v2}$ il consumo energetico dato dalla trasmissione
	- $\psi_{v3}$ la memoria disponibile nel nodo
- Dato un arco $e \in E_N$, una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia: 
	- $\epsilon_{e0}$ la banda di trasmissione dell'arco

Definiamo le variabili:
- $x_{ik}^a \in \{0, 1\}$ tc   $x_{ik}^a = 1$ sse $i \in V_M^a$ assegnato a $k \in V_N$
- $y_{mn}^a \in \{0, 1\}$ tc  $y_{mn}^a = 1$ sse $m \in E_M^a$ assegnato a $n \in E_N$

## Modello di Latenza

### Latenza di Calcolo
La latenza di calcolo sul server $k$ per richieste fatte alla variante $a$-esima la possiamo scrivere come:
$$T_k^{c-a} = \lambda^a \sum_{i \in V_M^a} \frac{\phi_{i0}}{\psi_{k0}} x_{ik}^a$$
Da cui il tempo di calcolo totale sul dispositivo $k$ è dato da:
$$T_k^c = \sum_{a \in A_{idx}} T_k^{c-a}$$

> [!TODO] Parallelismo
> Questa formulazione tiene conto del tempo di calcolo totale, ma non della latenza per l'inferenza di una singola richiesta perché questa dipende dall'eventuale parallelismo sul nodo del sistema. Si potrebbe assumere un tempo medio prendendo il numero di thread che eseguono richieste sul modello??

### Latenza di Trasmissione
La latenza di trasmissione dipende invece dalla somma delle latenze di trasmissione in funzione delle bande dei link fisici a cui i link logici sono assegnati. Per la variante $a$-esima sul server $k$:

$$T_{k}^{x-a} = \lambda^a \sum_{m = (i, j) \in E_M^a} \hspace{0.1cm} \sum_{n \in E_N \wedge k == n[0]} \frac{\eta_{m0}}{\epsilon_{n0}}y_{mn}^a$$
Da cui il tempo totale di trasmissione per il dispositivo $k$ è dato da:
$$T_k^x = \sum_{a \in A_{idx}} T_k^{x-a}$$

> [!TODO] Parallelismo
> Anche in questo caso forse va considerato il parallelismo di esecuzione e di trasmissione.


### Latenza totale
La latenza totale è quindi data da
$$T = \sum_{k \in E_N} (T_k^c + T_k^x)$$

## Modello di Energia

### Energia di Calcolo
L'energia data dal calcolo per $k \in V_N$ è data da:
$$E_k^c = T_k^c * \psi_{k1}$$


> [!NOTE] Parallelismo
> In questo caso anche se il parallelismo entra in gioco nel tempo, comunque devo considerare tutto il consumo.

### Energia di Trasmissione
L'energia data dalla trasmissione per $k \in V_N$ è data da:
$$E_k^x = T_k^x * \psi_{k2}$$
### Energia Totale
$$E = \sum_{k \in V_N} (E_k^x + E_k^c)$$

## Modello di Memoria
Dato un livello $v \in V_M^a$, la memoria da lui occupata è data da $\phi_{v1} + \phi_{v2}$. Assunto che la memoria allocata per l'output sia riusata per l'output dei livelli successivi, la memoria totale allocata per ogni nodo $k$ per la variante $a$ è data da:
$$M_k^a = \sum_{i \in V_N^a} (\phi_{i1}*x_{ik}^a) + \max_{i \in V_N^a}(\phi_{i2} * x_{ik}^a) $$
Che però porta ad una formulazione non lineare del problema.
Definiamo quindi una variabile ausiliaria $m_k^a$, definita in maniera tale che:
$$m_k^a \ge \phi_{i2} * x_{ik}^a \hspace{0.75cm} \forall i \in V_N^a$$

> [!TODO] Formulazione Lineare
> Controllare che questo vincolo sia sufficiente a dare una formulazione lineare


Da ciò quindi la memoria usata sul server $k$ dalla variante $a$-esima diventa:
$$M_k^a = \lambda^a * \left( \sum_{i \in V_N^a} \left( \phi_{i1}*x_{ik}^a \right) + m_k^a \right)$$

Da cui, la memoria totale necessaria sul singolo server $k$ è data da:
$$M_k = \sum_{a \in A_{idx}}  M_k^a$$

> [!TODO] Parallelismo
> Anche qui stiamo assumendo che le richieste vengano gestite in parallelo e che quindi vi sia una copia di modello per ogni richiesta. In generale bisogna capire bene l'aspetto del parallelismo e se/come inserirlo nella formulazione.

## Formulazione
Indichiamo con:
- $V_I^a \subset V_M^a$ il sottoinsieme di nodi di input della variante $a$; 
- $V_O^a \subset V_M^a$ il sottoinsieme di nodi di output della variante $a$

Siano:
- $\alpha_c$ e $\alpha_x$ i pesi della latenza di calcolo e di trasmissione
- $\beta_c$ e $\beta_x$ i pesi dell'energia di calcolo e di trasmissione

Sia $J_0$ l'energia massima consumabile sul server $0$ che fa partire l'inferenza.  

Tali che:
$$\sum_{i \in \{x,c\}} \alpha_i + \sum_{i \in \{x,c\}} \beta_i = 1$$

In definitiva il problema diventa il seguente:
$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \alpha_c*T^c + \alpha_x*T^x + \beta_c*E^c + \beta_x*E^x \\
\\
Vincoli:\\
\sum_{k \in V_N} x_{ik}^a = 1 \hspace{1cm}  \forall a \in A_{idx}, \forall i \in V_M^a \\
\sum_{n \in E_N} y_{mn}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall m \in E_M^a \\
x_{i0}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall i \in V_I^a \\ 
x_{j0}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall j \in V_O^a \\
\\

x_{ik}^a = \sum_{h \in V_N} y_{(i,j)(k,h)}^a \hspace{1cm} \forall a \in A_{idx}, \forall (i,j) \in E_M^a, \forall k \in V_N \\
x_{jh}^a = \sum_{k \in V_N} y_{(i,j)(k,h)}^a \hspace{1cm} \forall a \in A_{idx}, \forall (i,j) \in E_M^a, \forall h \in V_N \\
\\

M_k \le \psi_{k3} \hspace{1cm} \forall k \in V_N \\
m_k^a \ge \phi_{i2}^a * x_{ik}^a \hspace{0.75cm} \forall a \in A_{idx}, \forall i \in V_N^a \\
\\

J_0 \le E_0\\
\\

x_{ik}^a \in \{0, 1\} \hspace{1cm} \forall a \in A_{idx}, \forall i \in V_M^a, \forall k \in V_N \\
y_{mn}^a \in \{0, 1\} \hspace{1cm} \forall a \in A_{idx}, \forall m \in E_M^a, \forall n \in E_N \\

\end{matrix}

$$
Analisi vincoli:
1. Imporre che ogni nodo di ogni variante sia mappato su esattamente un server
2. Imporre che ogni arco di ogni variante sia mappato su esattamente un arco della rete di server
3. Imporre che almeno i livelli di input di ogni variante si trovino sul server 0 che fa partire l'inferenza (per assunzione)
4. Imporre che i livelli di output si trovino sul server 0 che fa partire l'inferenza (per assunzione)
5. Imporre rispetto flusso di invio: $i \in V_M^a$ è assegnato a $k \in V_N$, allora i dati prodotti da $i$ sono inviati ai server che hanno i nodi successori di $i$
6.  Imporre rispetto flusso di ricezione
7. Imporre il vincolo sulla memoria utilizzabile
8. Imporre che le variabili ausiliari per la memoria siano coerenti con il massimo
9. Imporre che il consumo energetico sul device non superi una soglia massima stabilita


> [!TODO] Dubbi Normalizzazione
> - Per quanto riguarda la normalizzazione delle parti della funzione obiettivo, bisogna introdurre nella modellazione o è più una cosa implementativa?
> - Per la normalizzazione del tempo devo dividere per il tempo massimo tra tutti i server (la somma dei rapporti che sono stati modellati)
> - Per la normalizzazione dell'energia devo fare la stessa cosa??


> [!NOTE] Consumo energetico device
> Anziché aggiungere il vincolo sul consumo del device, si potrebbe ritornare il consumo calcolato per il device come risultato dell'ottimizzazione. A quel punto è il device che decide, in base al suo stato di carica, quale soluzione usare. Stessa cosa si potrebbe fare per la latenza complessiva delle richieste.
