# Ottimizzazione 3.0 (Tentativo di Variante con Componenti)
In questa formulazione teniamo in conto anche la presenza delle componenti e il parallelismo che ne deriva.

Consideriamo un solo modello e una sola richiesta (in prima istanza).

Sia dato un modello rappresentato come un DAG $G_M = (V_M, E_M)$.
Sia dato un grafo di rete, non necessariamente un DAG $G_N = (V_N, E_N)$ 

Sia dato un $G_M \in A$. Definiamo:
- Dato un nodo $v \in V_M$, una tupla $\phi_v$ di caratteristiche del nodo del modello; in particolare sia:
	- $\phi_{v0}$ i flops del livello
	- $\phi_{v1}$ la memoria dovuta ai pesi del livello
	- $\phi_{v2}$ la memoria dovuta all'output
- Dato un arco $e \in E_M$, una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia:
	- $\eta_{e0}$ la size dei dati che caratterizzano l'arco

Consideriamo $G_N$. Definiamo:
- Dato un nodo $v \in V_N$, una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia:
	- $\psi_{v0}$ i flops al secondo calcolabili
	- $\psi_{v1}$ il consumo energetico dato dal calcolo
	- $\psi_{v2}$ il consumo energetico dato dalla trasmissione
	- $\psi_{v3}$ la memoria disponibile nel nodo
	- $\psi_{v4}$ il massimo numero di componenti parallele che il nodo può gestire
- Dato un arco $e \in E_N$, una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia: 
	- $\epsilon_{e0}$ la banda di trasmissione dell'arco
	- $\epsilon_{e1}$ la latenza di trasmissione dell'arco

Sia dato $G_C = (V_C, E_C)$ Il grafo delle componenti tale per cui:
- $V_C = \{1, 2, ..., |V_C|\}$
- $E_C$ è una mesh delle componenti tale per cui $\nexists (p, p) \in E_C$, ovvero non esistono cappi nel grafo delle componenti 

Il nostro obiettivo è duplice:
- Da un lato vogliamo mappare i livelli $i \in V_M$ sulle componenti $p \in V_C$
- Dall'altro vogliamo mappare le componenti $p \in V_C$ sui nodi della rete $k \in V_N$

Definiamo le seguenti variabili di base:
- $x_{ip} = 1$ sse $i \in V_M$ viene messo in $p \in V_C$
- $y_{pk} = 1$ sse $p \in V_C$ viene mappato su $k \in V_N$
- $w_{mc} = 1$ sse $m \in E_M$ viene mappato su $c \in E_C$
- $z_{cn} = 1$ sse $c \in E_C$ viene mappato su $n \in E_N$
- $a_{pq} = 1$ sse $p \in V_C$ passa dati a $q \in V_C$ e $p \neq q$


## Tempo
Definiamo:
- $f_k(i)$ la funzione che dato il livello $i \in V_M$ definisce il tempo di calcolo di quel livello sul server $k \in V_N$
- $g_k(m, h)$ la funzione che dato l'arco $m \in E_M$ e l'arco $(k, h) \in E_N$ definisce il tempo di trasmissione dei dati dell'arco $m$ da sull'arco $(k,h)$.

### Tempo di Calcolo
Il tempo di calcolo di una componente $p \in V_C$ dipende dai nodi assegnati alla componente e dal server a ci la componente è assegnata; in particolare se assegnata al server $k \in V_N$ abbiamo: 
$$
T_{pk}^c = \sum_{i \in V_M} f_k(i) \cdot b_{ipk}
$$
Dove definiamo la variabile $b_{ipk} = x_{ip} \cdot y_{pk}$ soggetta ai vincoli:
1. $b_{ipk} \le x_{ip}$
2. $b_{ipk} \le y_{pk}$
3. $b_{ipk} \ge x_{ip} + y_{pk} - 1$
Tale per cui $b_{ipk} = 1$ se $i$ è assegnato a $p$ e $p$ è assegnato a $k$


Il tempo di calcolo di una componente è dato quindi da:
$$
T_p^c = \sum_{k \in V_N} T_{pk}^c
$$
Essendo infatti la componente assegnata ad un solo server vi sarà un unico termine non nullo della sommatoria.

### Tempo di Trasmissione
Dato un livello $i \in V_M$ dato alla componente $p \in V_C$, questo livello dovrà trasmettere e componenti ai nodi successivi che non si trovano nella stessa componente, ovvero tali per cui l'arco $(i,j) \in E_M$ fa parte del cut del grafo. Da ciò possiamo dire che il tempo di trasmissione per una componente è dato da:
$$
T^x_{pk} = \sum_{m \in E_M} \hspace{0.1cm} \sum_{n \in V_N \wedge k = n[0]} \hspace{0.1cm} \sum_{c \in E_C \wedge p=c[0]} g_k(m, h) \cdot o_{mcn}
$$
Dove definiamo $o_{mcn} = w_{mc} \cdot z_{cn}$ soggetta ai vincoli:
1. $o_{mcn} \le w_{mc}$
2. $o_{mcn} \le z_{cn}$
3. $o_{mcn} \ge w_{mc} + z_{cn} - 1$

Il tempo di trasmissione di una componente è dato quindi da:
$$
T_p^x = \sum_{k \in V_N} T_{pk}^x
$$
Notiamo che il tempo di trasmissione dei dati all'interno della singola componente viene modellato come il tempo di trasmissione dei dati da una componente a se stessa.
### Ritardo
Data una componente $p \in V_C$ definiamo $P(p)$ l'insieme delle componenti che gli passano il suo input; allora il ritardo nel calcolo di una componente lo possiamo definire come:

$$
D_p = \max_{q \in P(p)}{D_q} + T_p^c + T_p^x
$$
Definiamo una componente detta $out$ tale che riceve l'output della computazione complessiva; allora $D_{out}$ è il ritardo di calcolo della singola richiesta, ovvero:
$$
D_{out} = \max_{q \in P(out)}{D_q}
$$


#### Rimozione del massimo
Definiamo $D_{p-prev} = \max_{q \in P(p)}{D_q}$, allora deve valere il seguente vincolo:
$$D_{p-prev} \ge D_q \cdot a_{qp} \hspace{0.5cm} \forall \hspace{0.1cm} q \neq p$$
Allo stesso tempo dobbiamo rimuover il prodotto tra le variabili e questo possiamo farlo introducendo la variabile $D_{qp}$ tale che:
- $D_{qp} \le D_q$
- $D_{qp} \le M \cdot a_{qp}$
- $D_{qp} \ge 0$

Da cui abbiamo:
$$
D_{p-prev} \ge D_{qp} \hspace{0.5cm} \forall q \neq p
$$

Questa formulazione garantisce che:
- Se $a_{qp} = 1$, cioè $p$ dipende da $q$, allora $D_{p-prev}$ è almeno quando $D_{q}$
- Se $a_{qp} = 0$, cioè $p$ NON dipende da $q$, allora non ci sono vincoli particolari imposti su $D_{p-prev}$

#### Costo del ritardo
Definiamo $\mu_d$ il termine di normalizzazione per il ritardo; il costo del ritardo è definito come:
$$
C_d = \frac{D_{out}}{\mu_d}
$$

## Energia
Definiamo le funzioni:
- $r_k(i)$ che definisce l'energia consumata per il calcolo del livello $i \in V_M$ su server $k \in V_N$
- $s_k(m,h)$ che definisce l'energia di trasmissione per i dati dell'arco $m \in E_M$ sull'arco $(k,h) \in E_N$

### Energia di Calcolo
L'energia per il calcolo di una componente $p \in V_C$ su server $k \in V_N$ è data da:
$$
E_{pk}^c = \sum_{i \in V_M} r_k(i) \cdot b_{ikp}
$$
L'energia consumata per il calcolo della componente $p \in V_C$ è quindi data da:
$$
E_p^c = \sum_{k \in V_N} E_{pk}^c
$$

### Energia di Trasmissione
Energia di trasmissione da una componente $p \in V_C$ assegnata al server $k \in V_N$ è dato da:
$$
E^x_{kp} = \sum_{m=(i,j) \in E_M} \hspace{0.1cm} \sum_{n \in V_N \wedge k = n[0]} \hspace{0.1cm} \sum_{c \in E_C \wedge p=c[0]} s_k(m, h) \cdot o_{mcn}
$$
Energia di trasmissione di una componente è dato quindi da:
$$
E_p^x = \sum_{k \in V_N} E_{kp}^x
$$

### Energia del Device
Dato un device $k \in V_N$; il consumo energetico complessivo sul device è dato da:
$$J_k = J_k^x + J_k^c$$

Dove abbiamo che :
$$
J_k^c = \sum_{p \in V_C} E_{pk}^c
$$
$$
J_k^x = \sum_{p \in V_C} E_{pk}^x
$$

Ovvero il consumo complessivo è dato dal consumo di calcolo e di trasmissione di tutte le componenti assegnate al device.

### Consumo Energetico Complessivo
Il consumo energetico complessivo è quindi dato da:

$$E = \sum_{p \in V_C} \left( E_p^c + E_p^x \right)$$


#### Costo dell'Energia
Definito $\mu_e$ il termine di normalizzazione per l'energia, il costo dato dal consumo energetico è dato da:
$$C_e = \frac{E}{\mu_e}$$

## Memoria
RIVEDERE LA MODELLAZIONE DELLA MEMORIA


## Vincoli di Assegnazione
I vincoli di assegnazione sono i seguenti:
$$
\begin{matrix}
\sum_{p \in V_C} x_{ip} = 1 \hspace{0.5cm} \forall i \in V_M \\
\\
\sum_{k \in V_N} y_{pk} = 1 \hspace{0.5cm} \forall p \in V_C \\
\\
\sum_{c \in E_C} w_{mc} = 1 \hspace{0.5cm} \forall m \in E_M \\
\\
\sum_{n \in E_N} z_{cn} = 1 \hspace{0.5cm} \forall c \in E_C \\


\end{matrix}
$$
Dove abbiamo che:
- Il primo vincolo impone che ogni nodo sia assegnato ad una componente
- Il secondo gruppo di vincoli impone che una componente sia sempre assegnata ad un server
- Il terzo vincolo impone che un arco del modello sia mappato su un unico arco tra componenti
- Il quarto vincolo impone che un passaggio di dati tra componenti sia mappato su un passaggio di dati tra server

Si noti che in questa formulazione non ci sono vincoli che impongano che ad una componente siano assegnati livelli: una componente potrebbe tranquillamente essere vuota e assegnata al server; semplicemente in quel caso non entra in gioco all'interno del grafo quoziente.

## Vincoli di Flusso
Ci sono due tipi di vincoli di flusso da considerare:
- Riguardante il flusso nel mapping livelli su componenti
- Riguardante il flusso nel mapping componenti su nodi

Il vincolo di flusso per l'invio tra componenti è dato da:
$$
x_{ip} = \sum_{q \in V_C} w_{(i,j)(p,q)} \hspace{0.5cm} \forall (i,j) \in E_M, \forall p \in V_C
$$
Il vincolo di flusso per la ricezione tra componenti è dato da:
$$
x_{jq} = \sum_{p \in V_C}{w_{(i,j)(p,q)}} \hspace{0.5cm} \forall (i,j) \in E_M, \forall q \in V_C
$$



Il vincolo di flusso per l'invio tra server è dato da:
$$
y_{pk} = \sum_{h \in V_N}{z_{(p,q)(k,h)}} \hspace{0.5cm} \forall (p,q) \in E_C, \forall k \in V_N
$$
Il vincolo di flusso per la ricezione tra server diversi è dato da:
$$
y_{qh} = \sum_{k \in V_N}{z_{(p,q)(k,h)}} \hspace{0.5cm} \forall (p,q) \in E_C, \forall h \in V_N
$$

## Vincoli di Aciclicità
(Dal paper sull'aciclicità)
Vogliamo che il grafo quoziente, ovvero il grafo risultante dal taglio sia aciclico.

Le variabili $a_{pq}$  rappresentano in sostanza una entry della matrice di adiacenza $A$ del grafo quoziente in cui una entry è uno se c'è un passaggio di dati tra le due componenti

Abbiamo allora che:
$$
x_{ip} + x_{jq} - 1 \le a_{pq} \hspace{0.5cm} \forall (i,j) \in E_M, p \neq q, p \in V_C, q \in V_C
$$
Notiamo che qui il valore 1 è indotto solo e unicamente se i nodi $i$ e $j$ tra cui esiste un arco del modello si trovano in componenti diverse.

Imponiamo che la matrice di adiacenza sia strettamente triangolare superiore:
$$
a_{pq} = 0 \hspace{0.5cm} \forall p \ge q, q \in V_C, p \in V_C
$$
In questo modo il flusso di dati può solo andare avanti tra le componenti e mai tornare indietro, impedendo la creazione di qualsiasi tipo di ciclo.


$$
	a_{pq} \ge \frac{1}{n} \sum_{(i,j) \in E_M} y_{(i,j)(p,q)}
$$


## Vincoli di Coerenza Aggiuntivi
Definiamo le due componenti $in$ e $out$ tali che $in \in V_C$ e $out \in V_C$; queste componenti rappresentano rispettivamente la sorgente dei dati e il sink dei risultati. Saranno tali per cui:
- Alla componente $in$ sono assegnati solo i nodi sorgente
- Alla componente $out$ sono assegnati solo i nodi di output
- Entrambe le componenti sono assegnate al device che fa partire l'inferenza (assunto il device ad indice $0$)

Definiamo due nodi:
- $gen \in V_M$ che rappresenta il nodo che genera tutti gli input del modello
- $rcv \in V_M$ che rappresenta il nodo che riceve tutti gli output del modello


Per cui possiamo scrivere le seguenti:
- $x_{gen,in} = 1$
- $x_{rcv,out} = 1$
- $y_{in,0} = 1$
- $y_{out,0} = 1$

Inoltre, per comodità di gestione, imponiamo che le componenti di $in$ e $out$ abbiano solo rispettivamente i nodi $gen$ e $rcv$, ovvero:
$$
\sum_{i \in V_M} x_{i,in} = 1
$$
$$
\sum_{i \in V_M} x_{i,out} = 1
$$


Questo garantisce la presenza dell'input e dell'output a livello del device.

Attenzione! Nell'ordinamento degli indici delle componenti, l'ordinamento deve essere fatto in modo che $in$ sia sempre il primo e $out$ sia sempre l'ultimo: si potrebbero usare ad esempio indici -1 e $|V_C| + 1$


## Definizione del Problema
Rivedere i vincoli su posizione di input ed output


$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \omega_d*C_d + \omega_e*C_e\\
\\
Vincoli:\\
Sopra

\end{matrix}

$$


## Considerazioni
La formulazione è sicuramente più complessa rispetto al semplice mapping di livelli sui server, ma permette di tenere in considerazione l'eventuale parallelismo tra componenti che sono in esecuzione sullo stesso server o tra server diversi, cosa che prima non era presa in considerazione.

L'aumento della complessità è sicuramente un fattore limitante che rende il problema potenzialmente difficile da risolvere. Si potrebbe quindi pensare ad una forma di minimizzazione alternata in cui si risolve prima un problema e poi l'altro:
1. Init casuale assegnazione componenti ai server
2. Minimizzo per l'assegnazione dei livelli alle componenti
3. Minimizzo per l'assegnazione delle componenti ai server
Procedo tra 2 e 3 fino a convergenza e/o tempo limite (dipende molto da quanto il solver ci mette).
Questo aspetto è evidente ad esempio nelle assegnazioni livelli, componenti, server e nella formulazione del tempo di calcolo:
$$
T_{p}^c = \sum_{k \in V_N} \sum_{i \in V_M} f_k(i) \cdot x_{ip} \cdot y_{pk} = \sum_{k \in V_N} y_{pk} \sum_{i \in V_N} f_k(i) \cdot x_{ip}
$$
In questa formulazione infatti una volta che si impostano le assegnazioni $y_{pk}$, il problema si semplifica molto (viceversa scambiando le sommatorie).

Per quanto riguarda il numero di componenti, si può imporre che non superi un certo massimo che i vari server "dichiarano"; il problema è sicuramente che deve essere stabilito a priori, proprio come nel k-means.

Questa formulazione ha però diversi vantaggi:
1. Permette di tenere in conto l'impatto dell'esecuzione parallela
2. Definire il ritardo nella risposta in maniera più accurata
3. Specificare dei vincoli sul ritardo
4. Tenere in conto il partizionamento già nella fase di risoluzione del problema permettendo un'ottimizzazione conseguente

### Separazione del Problema
Assumiamo di fare una minimizzazione alternata.

Inizializziamo l'assegnazione delle componenti tutte quante sul server 0 (per assunzione il device che fa partire l'inferenza).

Allora possiamo scrivere che:
$$
T_p^c = \sum_{i \in V_M} x_{ip} \sum_{k \in V_N} f_k(i) \cdot y_{pk} 
$$
$$
T^x_{p} = \sum_{k \in V_N} \hspace{0.1cm} \sum_{m \in E_M} \hspace{0.1cm} \sum_{c \in E_C \wedge p=c[0]} w_{mc} \sum_{n \in V_N \wedge k = n[0]} \hspace{0.1cm} g_k(m,h) \cdot z_{cn}
$$


#### Minimizzazione $V_M \rightarrow V_C$ fissato $V_C \rightarrow V_N$ (i % 2 == 0)
Se fissiamo il mapping $V_C \rightarrow V_N$, allora le variabili $y_{pk}$ e $z_{cn}$ sono fissate; 

Questo ci permette di rimuovere i vincoli di flusso riguardanti queste variabili.

Rimangono invece:
- Vincoli di flusso riguardanti $x_{ip}$ e $w_{mc}$
- Vincoli di aciclicità

#### Minimizzazione  $V_C \rightarrow V_N$ fissato $V_M \rightarrow V_C$ (i % 2 == 1)
Se fissiamo il mapping $V_M \rightarrow V_C$, allora le variabili $x_{ip}$ e $w_{mc}$ sono fissate; 
Questo ci permette di rimuovere i vincoli di flusso riguardanti queste variabili.

Rimangono invece:
- Vincoli di flusso riguardanti $y_{pk}$ e $z_{cn}$