## Conversioni

Si può convertire direttamente un modello Keras in un modello LiteRT usando il converter di TF. Il problema è che la conversionde del modello cos' com'è solleva un errore in conversione: questo è dovuto al fatto che non tutte le operazioni TF sono supportate da TFLite (ad esempio i Conv2D non sono supportati).
![[Schermata del 2024-12-14 09-29-28.png|Errore di Compatibilità delle Operazioni]]


Per supportare queste operazioni si può abilitare l'integrazione della definizione delle operazioni non supportate in TFLite; riferimenti a:
- Integrazione altre operazioni
	- https://ai.google.dev/edge/litert/models/ops_select
	-  https://ai.google.dev/edge/litert/models/op_select_allowlist.md
- Operatori supportati nativamente e relazioni con altri operatori
	- https://ai.google.dev/edge/litert/models/ops_compatibility.md

Con il blocco di codice seguente si possono integrare le operazioni TF nel modello LiteRT.
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```


> [!Warning] Dimensione modello
> Capire se solo le operazioni mancanti sono aggiunte alla definizione del modello o se lo sono tutte (dal nome si parla di SELECT, quindi probabilmente solo quelle che mancano sono inserite).


Dopo questa integrazione il modello viene creato correttamente, ma c'è ancora un warning che viene sollevato sulla mancanza di un'operazione, in particolare su FlexConv2D.
![[Schermata del 2024-12-14 09-56-31.png|Warning su FlexConv2D]]

### Segnature del modello
Per quanto riguarda le output signature sono mantenute uguali a quelle del modello keras:
![[Schermata del 2024-12-14 18-54-10.png|Output Signature uguale a Keras]]


Il problema sono le input signature: la chiave non è presa considerando il nome dato all'input del dizionario all'atto della creazione in keras, ma il nome del livello di input!! Bisogna cambiare il nome del livello all'atto della conversione del modello...
![[Schermata del 2024-12-14 18-56-46.png|Input Signature con stesso nome livello di input - CLONE in Append]]


Una volta aggiunto il blocco per il cambio nome come riportato in [[Split del Modello]] il problema si risolve: questo permette di raccordare l'output dato da un sotto modello in TFLite e da un sotto modello eseguito direttamente in Keras.
![[Schermata del 2024-12-14 19-14-19.png|Input Signature con stesso nome livello di input - NO CLONE in Append]]

Il problema sull'input rimane per il primo input al modello...

## Inferenza
Sembra esserci un'incompatibilità tra versioni di numpy usate se non se ne specifica una:
![[Schermata del 2024-12-14 16-39-10.png|Incompatibilità tra versioni di Numpy]]

Per risolvere il problema facciamo il downgrade ad una versione di numpy precedente alla 2.0 eseguendo; `pip install "numpy<2"`. Questo sembra garantire il funzionamento sia di questo che delle altre cose implementate.

La SignatureRunner di LiteRT non accetta parametri senza che gli sia associato nome:
![[Schermata del 2024-12-14 16-47-24.png|Errore sugli argomenti]]


`SignatureRunner.get_input_details()` permette di ottenere info sui parametri che il runner si aspetta, tra cui il nome di questo parametro.


Gli output ottenuti con Yolo sembrano diversi, anche se la norma della differenza è abbastanza bassa; il motivo potrebbe essere dovuto alle conversioni fatte internamente quando viene costruito il nuovo modello TFLite.
![[Schermata del 2024-12-14 17-45-05.png|Output diversi su Yolo]]


Stessa cosa succede con MobileNetV3Large: la differenza è molto piccola e la classe predetta è la stessa.
![[Schermata del 2024-12-14 18-16-21.png|Output Diversi su MobileNetV3Large]]


L'uso di `allocate_tensor()` sembra cambiare l'errore sui tensori prodotti in output: in figura l'esempio con MobileNetV3Large.

| Senza allocate_tensor   | ![[Schermata del 2024-12-14 18-24-21.png]] |
| ----------------------- | ------------------------------------------ |
| **Con allocate_tensor** | ![[Schermata del 2024-12-14 18-25-04.png]] |

La segnatura del modello solleva il seguente warning:
![[Schermata del 2024-12-16 11-46-00.png|Warning segnatura]]

Per risolverlo, impostiamo la *input_signature* come un dict: in questo modo la funzione di call può ricevere l'input nella maniera che si aspetta (che è un dizionario infatti). Per fare questo non si può fare la conversione direttamente da un keras model, ma bisogna passare per forza da un *saved_model* di cui si imposta la input signature (vedere codice successivo).
```python
inputs = {}
for key in kerasModel.input:
	inpTens = kerasModel.input[key]
	inputs[key] = tf.TensorSpec(
		shape=inpTens.shape, dtype=inpTens.dtype, name=inpTens.name
	)

archive.add_endpoint(
	name="serve",
	fn=kerasModel.call,
	input_signature=[inputs],
)

archive.write_out(f"./models/{kerasModel.name}")
```

Facendo in questo modo il warning scompare e la segnatura adesso è sia posizionale sia con le keyword.
![[Schermata del 2024-12-16 11-52-02.png|Segnatura della *serve*]]


> [!Warning] 
> La prima funzione segnata con la `add_endpoint` è segnata come `serving_default` e deve essere acceduta con quella.


## Soluzione Mista
La soluzione mista sembra funzionare bene con un errore basso.
![[Schermata del 2024-12-16 08-21-30.png|Errore della soluzione Mista]]


Un'accortezza da adottare è relativa ai tipi diversi dati in output dalle due inferenze:
- Keras dà in output un tensore TensorFlow
- LiteRT dà in output un tensore numpy
L'errore seguente dice che non si possono mischiare i due tipi di tensore nell'input. Bisogna quindi avere uniformità nei tipi che si passano in input. Si possono quindi passare o solo dei tensori TF oppure solo dei tensori numpy (per adesso andiamo con i tensori numpy che sono un tipo di livello più basso)

![[Schermata del 2024-12-16 08-23-35.png|Hello World]]


## Quantizzazione
L'abilitazione delle quantizzazioni sembra essere abbastanza facile: è sufficiente aggiungere un attributo al converter.

Riferimento https://ai.google.dev/edge/litert/models/post_training_quantization
![[Schermata del 2024-12-23 11-32-59.png|Tipi di Quantizzazione Offerti]]
### Dynamic

> [!Quote] Title
> Dynamic range quantization provides reduced memory usage and faster computation without you having to provide a representative dataset for calibration. This type of quantization, statically quantizes only the weights from floating point to integer at conversion time, which provides 8-bits of precision.
> 
> To further reduce latency during inference, "dynamic-range" operators dynamically quantize activations based on their range to 8-bits and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully fixed-point inferences. However, the outputs are still stored using floating point so the increased speed of dynamic-range ops is less than a full fixed-point computation.

In questo caso quindi:
- Parametri salvati come interi a 8bit
- Attivazioni convertite a precisione ad 8 bit

### Full Integer
Servono delle inferenze di prova per calibrare al meglio i pesi che vengono usati.

### Float16

> [!Quote] 
> The advantages of float16 quantization are as follows:
>- It reduces model size by up to half (since all weights become half of their original size).
>- It causes minimal loss in accuracy.
>- It supports some delegates (e.g. the GPU delegate) which can operate directly on float16 data, resulting in faster execution than float32 computations.
>
>The disadvantages of float16 quantization are as follows:
>- It does not reduce latency as much as a quantization to fixed point math.
> - By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU. (Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

https://ai.google.dev/edge/litert/models/post_training_float16_quant

> [!Quote] 
> LiteRT supports converting weights to 16-bit floating point values during model conversion from TensorFlow to LiteRT's flat buffer format. This results in a 2x reduction in model size. 
> 
> Some hardware, like GPUs, can compute natively in this reduced precision arithmetic, realizing a speedup over traditional floating point execution. The LiteRT GPU delegate can be configured to run in this way. 
> 
> However, a model converted to float16 weights can still run on the CPU without additional modification: the float16 weights are upsampled to float32 prior to the first inference. This permits a significant reduction in model size in exchange for a minimal impacts to latency and accuracy.

Quindi questa quantizzazione è significativa in termini di latenza solo se si ha a disposizione una GPU che supporta Float16; negli altri casi si ritorna a Float32.


### Risultati
Per quanto riguarda il tempo di conversione, la conversione a Full Integer richiede un tempo maggiore, probabilmente a causa della calibrazione.

Precisione e tempi di esecuzione (in locale) al variare delle quantizzazioni. Notiamo che le quantizzazioni che apportano maggiore beneficio, a scapito di una certa approssimazione, sono la Dynamic e la Full Integer; La Float16 non sembra dare beneficio in termini di tempo, ma questo è probabilmente dovuto alla mancata configurazione della GPU.

|                                   | No Quant    | Dynamic    | Full Integer | Float16     |
| --------------------------------- | ----------- | ---------- | ------------ | ----------- |
| Norma Infinito Differenza         | 1.09896e-07 | 0.00344    | 0.006...     | 8.72351e-05 |
| Tempo di Esecuzione (ns) (25 Run) | 3469235878  | 2526344940 | 2535534742   | 3607644266  |

Un'altra cosa interessante è la possibilità di mescolare i diversi tipi di quantizzazione. Di seguito il test eseguito su un modello Yolo con:
- Massimo numero di livelli per partizione pari a 20
- Tre tipi di quantizzazioni alternate: Nessuna, Dynamic e Float16
Otteniamo un risultato per la norma infinito della differenza di circa 0.00197058. Vediamo che l'errore è più basso del solo Dynamic, probabilmente perché l'effetto delle parti calcolate con dynamic è attenuato dalle altre quantizzazioni
![[Schermata del 2024-12-23 12-46-37.png|Errore con 3 Varianti di Quantizzazione]]

Eseguendo invece con due tipi di quantizzazione, alternando modello non quantizzato e modello con Float16, otteniamo il seguente errore. Anche in questo caso è più basso della sola quantizzazione Float16 (in effetti è circa la metà, cosa che ha senso visto che alterniamo con la variante senza quantizzazione).
![[Schermata del 2024-12-23 12-50-57.png|Errore con 2 Varianti di Quantizzazione]]