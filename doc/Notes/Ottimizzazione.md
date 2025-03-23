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

Dato un $a \in A_{idx}$, definiamo:
- $\Lambda^a$ come il numero di richieste fatte al sistema nell'unità di tempo per la variante $a$-esima;
- $\Lambda = \sum_{a \in A_{idx}} \Lambda^a$ Il numero totale di richieste fatte al sistema 
- $\lambda^a = \frac{\Lambda^a}{\Lambda}$ Il tasso di richieste fatto per la variante $a$-esima

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
	- $\epsilon_{e1}$ la latenza di trasmissione dell'arco

Dato una variante $G_M^a$ definiamo una tupla $\sigma^a$ di dimensione $|V_N|$, definita come:
- $\sigma_k^a$ rappresenta lo speedup di esecuzione del modello $a$-esimo sul server $k$-esimo
Lo speedup è inteso come speedup di un modello quantizzato rispetto al modello di base. Di conseguenza avremo:
- $\sigma_k^a = 1 \hspace{0.5cm} \forall k \in V_n$ se il modello $a$-esimo non è quantizzato
- $\sigma_k^a$ che può assumere un valore dipendente dalla profilazione del modello $a$-esimo quantizzato sul nodo $k$-esimo

Definiamo le variabili:
- $x_{ik}^a \in \{0, 1\}$ tc   $x_{ik}^a = 1$ sse $i \in V_M^a$ assegnato a $k \in V_N$
- $y_{mn}^a \in \{0, 1\}$ tc  $y_{mn}^a = 1$ sse $m \in E_M^a$ assegnato a $n \in E_N$

## Modello di Tempo

### Tempo di Calcolo
Il tempo di calcolo sul server $k$ per richieste fatte alla variante $a$-esima la possiamo scrivere come:
$$T_k^{c-a} =  \sum_{i \in V_M^a} \frac{\phi_{i0}}{\psi_{k0} * \sigma_k^a} x_{ik}^a$$
Definiamo il tempo massimo di calcolo tra i vari nodi della variante $a$-esima sul server $k$ come:
$$t_k^{c-a} = \max_{i \in V_M^a} \frac{\phi_{i0}}{\psi_{k_0} * \sigma_k^a}$$

Il tempo di calcolo totale per una singola richiesta per la variante $a$-esima è definito come:
$$T^{c-a} = \sum_{k \in V_N} T_k^{c-a}$$
Definiamo il massimo dei tempi di calcolo dei nodi della variante $a$-esima come :
$$t^{c-a} = \max_{k \in V_N} t^{c-a}_k$$

### Tempo di Trasmissione
La latenza di trasmissione dipende invece dalla somma dei tempi di trasmissione in funzione delle bande dei link fisici a cui i link logici sono assegnati. Per la variante $a$-esima sul server $k$:

$$T_{k}^{x-a} =  \sum_{m = (i, j) \in E_M^a} \hspace{0.1cm} \sum_{n \in E_N \wedge k == n[0]} \left( \frac{\eta_{m0}}{\epsilon_{n0}} + \epsilon_{n1} \right) * y_{mn}^a$$
Definiamo il tempo massimo per la trasmissione tra i nodi della variante $a$-esima come:
$$t^{x-a}_k = \max_{m \in E_M^a, n \in E_N \wedge k == n[0]} \left( \frac{\eta_{m0}}{\epsilon_{n0}} + \epsilon_{n1} \right)$$

Il tempo di trasmissione totale per singola richiesta sulla variante $a$-esima è dato da:
$$T^{x-a} = \sum_{k \in V_N} T_k^{x-a}$$
Il tempo di trasmissione massimo per singola richiesta ala variante $a$-esima è dato da:
$$t^{x-a} = \max_{k \in V_N} t_k^{x-a}$$

### Costo del tempo 
Definiamo $\mu_t^a = \max \{t^{x-a}, t^{c-a}\}$ il termine di normalizzazione per la latenza del modello $a$-esimo
Il costo dato dal tempo di inferenza del modello $a$-esimo è dato da:
$$C^a_t = \lambda^a \frac{T^{c-a} + T^{x-a}}{\mu_t^a}$$
Da cui il costo del tempo di inferenza complessivo è definito da:
$$C_t = \sum_{a \in A_{idx}} C_t^a$$

## Modello di Energia

### Energia di Calcolo
L'energia data dal calcolo della variante $a$-esima sul server $k$ è definita come:
$$E_k^{c-a} = T_k^{c-a} * \psi_{k1}$$
Il consumo energetico maggiore per la singola inferenza del modello $a$ sul server $k$ è dato da:
$$e_k^{c-a} = t_k^{c-a} * \psi_{k1}$$

Il consumo energetico complessivo dato dal calcolo della variante $a$-esima è dato da 
$$E^{c-a} = \sum_{k \in V_N} E_k^{c-a}$$
Mentre il consumo massimo è dato da:
$$e^{c-a} = \max_{k \in V_N} e_k^{c-a}$$


### Energia di Trasmissione
L'energia data dal calcolo della variante $a$-esima sul server $k$ è definita come:
$$E_k^{x-a} = T_k^{x-a} * \psi_{k2}$$
Il consumo energetico maggiore per la singola inferenza del modello $a$ sul server $k$ è dato da:
$$e_k^{x-a} = t_k^{x-a} * \psi_{k2}$$

Il consumo energetico complessivo dato dal calcolo della variante $a$-esima è dato da 
$$E^{x-a} = \sum_{k \in V_N} E_k^{x-a}$$
Mentre il consumo massimo è dato da:
$$e^{x-a} = \max_{k \in V_N} e_k^{x-a}$$


### Costo dell'energia
Definiamo $\mu_e^a = \max \{e^{x-a}, e^{c-a}\}$ il termine di normalizzazione per l'energia del modello $a$-esimo
Il costo di inferenza del modello $a$-esimo è dato da:
$$C^a_t = \lambda^a \frac{E^{c-a} + E^{x-a}}{\mu_e^a}$$
Da cui il costo dell'energia per l'inferenza è dato da:
$$C_t = \sum_{a \in A_{idx}} C_e^a$$
### Energia del device
Sia il device il server ad indice 0 in $V_N$.
La quantità di energia consumata da questo nodo di rete è data da:
$$E_0 = \sum_{a \in A_{idx}} \left( E_0^{x-a} + E_0^{c-a} \right)$$


## Modello di Memoria
Dato un livello $v \in V_M^a$, la memoria da lui occupata è data da $\phi_{v1} + \phi_{v2}$. Assunto che la memoria allocata per l'output sia riusata per l'output dei livelli successivi, la memoria totale allocata per ogni nodo $k$ per la variante $a$ è data da:
$$M_k^a = \sum_{i \in V_N^a} (\phi_{i1}*x_{ik}^a) + \max_{i \in V_N^a}(\phi_{i2} * x_{ik}^a) $$
Che però porta ad una formulazione non lineare del problema.
Definiamo quindi una variabile ausiliaria $m_k^a$, definita in maniera tale che:
$$m_k^a \ge \phi_{i2} * x_{ik}^a \hspace{0.75cm} \forall i \in V_N^a$$



Da ciò quindi la memoria usata sul server $k$ dalla variante $a$-esima diventa:
$$M_k^a = \sum_{i \in V_N^a} \left( \phi_{i1}*x_{ik}^a \right)  + m_k^a $$

Da cui, la memoria totale necessaria sul singolo server $k$ è data da:
$$M_k = \sum_{a \in A_{idx}}  M_k^a$$

> [!Note] Parallelismo
> Stiamo assumendo che per ogni variante, vi sia una sola copia attiva, quindi non c'è bisogno di moltiplicare per $\Lambda^a$ l'occupazione di memoria

## Formulazione
Indichiamo con:
- $V_I^a \subset V_M^a$ il sottoinsieme di nodi di input della variante $a$; 
- $V_O^a \subset V_M^a$ il sottoinsieme di nodi di output della variante $a$

Siano:
- $\omega_t$ il peso associato al tempo di inferenza
- $\omega_e$ il peso associato all'energia di inferenza

Sia $J_0$ l'energia massima consumabile sul device $0 \in V_N$.

Tali che:
$$ \omega_t + \omega_e = 1$$



In definitiva il problema diventa il seguente:
$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \omega_t*C_t + \omega_e*C_e\\
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

E_0 \le J_0\\
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


> [!NOTE] Consumo energetico device
> Anziché aggiungere il vincolo sul consumo del device, si potrebbe ritornare il consumo calcolato per il device come risultato dell'ottimizzazione. A quel punto è il device che decide, in base al suo stato di carica, quale soluzione usare. Stessa cosa si potrebbe fare per la latenza complessiva delle richieste.

> [!TODO] Parallelismo
> Questa formulazione tiene conto del tempo di calcolo totale, ma non della latenza per l'inferenza di una singola richiesta perché questa dipende dall'eventuale parallelismo sul nodo del sistema. Si potrebbe assumere un tempo medio prendendo il numero di thread che eseguono richieste sul modello??
> 
> Il parallelismo alla fine lo gestiamo così:
> - Data una coppia (server, sotto-modello), abbiamo un thread che si occupa dell'inferenza per quel sotto-modello
> - Il parallelismo di Onnx (o di altri engine di inferenza) è assunto uno nella modellazione, non necessariamente negli esperimenti


> [!Warning] Gestione della memoria
> La gestione della memoria potrebbe essere più complicata di così: dato un sottomodello c'è bisogno di mantenere tutti i suoi input in memoria prima di far partire l'inferenza su quel sotto modello. Questa cosa va modellata in qualche modo, altrimenti si perde la coerenza sulla memoria. Di fatto anche l'aspetto di riuso della memoria potrebbe non essere completamente verosimile perché (potrebbero) esserci più nodi in esecuzione contemporaneamente (parallelismo interno all'engine ignorato in teoria).
