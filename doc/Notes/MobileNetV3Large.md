## Analisi del Modello MobileNetV3Large

Analizzando l'attributo _config_ del modello (che restituisce un oggetto strutturato come un json che descrive il modello) sono riuscito a capire il problema relativo a quei livelli di Add e Multiply che, come mostrato in Netron, prendevano in input da un solo livello, cosa che risultava strana trattandosi di livelli aggreganti; il problema è che questi livelli in realtà prendono in input l'output di un livello precedente per poi eseguire un'operazione element-wise (appunto una somma o un prodotto) con una costante (nello specifico, per la somma la costante è sempre 3.0 mentre per la moltiplicazione è sempre 0.16 con 6 periodico). Quei livelli che venivano mostrati in netron in sostanza non sono affatto dei livelli (cioè istanziati in fase di costruzione del modello com[[2024-11-19]]e dei keras.Layer), ma sono il risultato di operazioni del seguente tipo:

```
x1 = prevLayer(prevPrevOutput)
x2 = x1 + 3
nextLayer(x2)
```

Quando viene fatta l'operazione di somma, questa non viene mappata su un oggetto di tipo layer all'atto della compilazione del modello, ma su un oggetto keras diverso, cioè un'istanza di una classe del modulo _keras.src.ops.numpy_ che mappano delle operazioni element-wise in keras.

Premettendo che non mi è molto chiaro il motivo per cui siano necessarie operazioni di questo tipo, in quanto riterrei più sensato imparare dei pesi del modello che fanno direttamente in modo, prendendo per riferimento l'esempio precedente, che x1 sia direttamente uguale al suo valore sommato 3 piuttosto che fare questo. Comunque, facendo il parsing del _config_, sono riuscito ad estrarre la classe di appartenenza di questi oggetti e farne il wrap all'interno di oggetti che si comportassero in modo simile a dei livelli, in modo da poterli distribuire al pari di livelli "classici".

> [!Warning] Validità e generalizzazione della soluzione
> A questo punto il dubbio che mi sorge è relativo alla validità di una soluzione di questo tipo in un contesto più generale: potrebbero esserci altri oggetti con cui keras fa cose simili di cui potrei non venire a sapere l'esistenza e che potrebbero dare problemi simili.

### Conferma di questo aspetto

In MobileNetV3 questo comportamento è dovuto alla definizione della _hard_sigmoid_ (https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/applications/mobilenet_v3.py#L538), definita come:

```python
def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)
```

Il motivo per cui si usa questa funzione è che approssima bene la funzione _swish_. Questa funzione _swish_, che si vuole usare come funzione di attivazione nel modello, è definita come:

$$
swish(x)=x * \sigma(x)
$$

e risulta pertanto abbastanza costosa dal punto di vista computazionale (https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa). La _hard_sigmoid_ permette di approssimarla (grafico) con un costo computazionale minore, che in questo tipo di modello è importante.
![[h-swish.png|Funzione H-Sigmoid e sua Approssimazione]]


Il motivo per cui non si fa imparare alla rete direttamente i pesi che permettono di ottenere quel risultato è dovuto al fatto che in questo caso stiamo approssimando la funzione di attivazione (per quanto l'obiezione rimane valida in parte).
