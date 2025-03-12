Le alternative di quantizzazione sono:
- LiteRT
- Onnx
- QKeras

# LiteRT
Le alternative di quantizzazione offerte da LiteRT sono:
- Dynamic
- Float16
- Full Integer
- 
![[Schermata del 2025-02-13 15-47-32.png|Quantizzazioni Offerte]]

https://ai.google.dev/edge/litert/models/model_optimization

## Dynamic
https://ai.google.dev/edge/litert/models/post_training_quant

> [!Quote] https://ai.google.dev/edge/litert/models/post_training_quant
> The activations are always stored in floating point. For ops that support quantized kernels, the activations are quantized to 8 bits of precision dynamically prior to processing and are de-quantized to float precision after processing. Depending on the model being converted, this can give a speedup over pure floating point computation.

Da quello che sembra emergere da qui, le attivazioni sono quantizzate e dequantizzate: questa potrebbe essere la causa della lentezza maggiore della dynamic quantization in alcuni casi.


> [!Quote] https://medium.com/better-ml/dynamic-quantization-fa2d64a24bfc
> As mentioned above dynamic quantization have the run-time overhead of quantizing activations on the fly. So, this is beneficial for situations where the model execution time is dominated by memory bandwidth than compute (where the overhead will be added). This is true for LSTM and Transformer type models with small batch size.

Da qui sembra che la quantizzazione dinamica in generale non sia adatta a casi non governati dalla latenza di memoria (che nel nostro non dovrebbe essere eccessiva). Inoltre la quantizzazione dinamica non darebbe benefici nemmeno in termini di trasmissione visto che input ed output di un sottomodello sarebbero riconvertiti prima della trasmissione: trasmetteremmo quindi dei tensori di float32 piuttosto che dei tensori di float16 o di int8.

Altra fonte utile:
https://www.datature.io/blog/introducing-post-training-quantization-feature-and-mechanics-explained


In questo caso l'utilità potrebbe quindi essere data dalla riduzione della dimensione del modello, ma per il resto il beneficio in termini di tempo di inferenza sembra nullo.

## Float16


> [!Quote] https://ai.google.dev/edge/litert/models/post_training_float16_quant
> LiteRT supports converting weights to 16-bit floating point values during model conversion from TensorFlow to LiteRT's flat buffer format. This results in a 2x reduction in model size. Some hardware, like GPUs, can compute natively in this reduced precision arithmetic, realizing a speedup over traditional floating point execution. The LiteRT GPU delegate can be configured to run in this way. However, a model converted to float16 weights can still run on the CPU without additional modification: the float16 weights are upsampled to float32 prior to the first inference. This permits a significant reduction in model size in exchange for a minimal impacts to latency and accuracy.

Da qui emerge che l'engine di esecuzione deve essere configurato per eseguire con float16


> [!Quote] https://ai.google.dev/edge/litert/models/post_training_float16_quant#convert_to_a_litert_model
> Finally, convert the model like usual. Note, by default the converted model will still use float input and outputs for invocation convenience.

Il modello anche se esportato con float16 comunque continua a prendere input ed output dei float normali: credo quindi che la cosa sia gestita internamente.


> [!quote] https://ai.google.dev/edge/litert/models/post_training_float16_quant
> It's also possible to evaluate the fp16 quantized model on the GPU. To perform all arithmetic with the reduced precision values, be sure to create the `TfLiteGPUDelegateOptions` struct in your app and set `precision_loss_allowed` to `1`, like this:

Bisogna quindi abilitare l'esecuzione su GPU usando i delegates. I delegate sono i driver di GPU che usa LiteRT per eseguire le operazioni su device.
```c
//Prepare GPU delegate.
const TfLiteGpuDelegateOptions options = {
  .metadata = NULL,
  .compile_options = {
    .precision_loss_allowed = 1,  // FP16
    .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
    .dynamic_batch_enabled = 0,   // Not fully functional yet
  },
};
```


### Delegates e Relativi

> [!quote] https://ai.google.dev/edge/litert/performance/delegates
> By default, LiteRT utilizes CPU kernels that are optimized for the [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) instruction set.
> ...
> Typically, you would have to write a lot of custom code to run a neural network through these interfaces. Things get even more complicated when you consider that each accelerator has its pros & cons and cannot execute every operation in a neural network. TensorFlow Lite's Delegate API solves this problem by acting as a bridge between the TFLite runtime and these lower-level APIs

> [!Quote] https://ai.google.dev/edge/litert/performance/delegates#delegates_by_platform
> Cross-platform (Android & iOS): **GPU delegate** - The GPU delegate can be used on both Android and iOS. It is optimized to run 32-bit and 16-bit float based models where a GPU is available. It also supports 8-bit quantized models and provides GPU performance on par with their float versions. For details on the GPU delegate, see [LiteRT on GPU](https://ai.google.dev/edge/litert/performance/gpu).


![[Schermata del 2025-02-13 19-02-19.png|Tabella Supporti Delegates]]

La figura mostra quali sono i supporti dei delegates: GPU li supporta tutti.


> [!Quote] https://github.com/tensorflow/tensorflow/issues/79111
> I apologize for the delayed response, As far I know currently TensorFlow python API does not support running stable delegates and If I'm not wrong we do support C++ API please refer this [official documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate), regarding stable delegate python API I'll confirm with relevant team if there is any future plan to support that and will update you here if I got any update from that relevant team
> https://github.com/tensorflow/tensorflow/issues/79111#issuecomment-2467789766
> 
> The stable delegate API is no longer experimental. But currently stable delegates are only supported for the C, C++, and Java APIs, not for the Python API (and I think also not for the C#, Objective-C, and Swift APIs).
> https://github.com/tensorflow/tensorflow/issues/79111#issuecomment-2489513987


Riguardo il flex delegate che a volte compare in esecuzione.
> [!Quote] https://docs.edgeimpulse.com/docs/tools/edge-impulse-for-linux/flex-delegates
> On Linux platforms without a GPU or neural accelerator your model is run using LiteRT (previously Tensorflow Lite). Not every model can be represented using native LiteRT (previously Tensorflow Lite) operators. For these models, 'Flex' ops are injected into the model. To run these models you'll need to have the flex delegate library installed on your Linux system. This is a shared library that you need to install once.

Sembra quindi che LiteRT non supporti l'esecuzione su GPU in automatico come la versione Tensorflow. Questa esecuzione va abilitata, ma il problema è che non sembrano esserci delle implementazioni di Delegates disponibili in Python.


> [!Quote] https://github.com/google/XNNPACK#xnnpack
> XNNPACK is a highly optimized solution for neural network inference on ARM, x86, WebAssembly, and RISC-V platforms. XNNPACK is not intended for direct use by deep learning practitioners and researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [TensorFlow Lite](https://www.tensorflow.org/lite), [TensorFlow.js](https://www.tensorflow.org/js), [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai), and [MediaPipe](https://mediapipe.dev).


#### PyCoral

> [!Quote] https://coral.ai/docs/edgetpu/tflite-python/#run-an-inference-with-the-pycoral-api
> The PyCoral API is a small set of convenience functions that initialize the TensorFlow Lite [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) with the Edge TPU delegate and perform other inferencing tasks such as parse a labels file, pre-process input tensors, and post-process output tensors for common models.

## Full Integer
To Check

## Test
Test su Istanza Cloud:
- Modello: ResNet152V2
- Macchina e2-standard-4
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi
![[Schermata del 2025-02-14 22-17-42.png]]

Test su Colab:
- Modello: ResNet152V2
- GPU Colab
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi
![[Schermata del 2025-02-15 15-20-37.png|Test Colab con GPU]]

Ad onor del vero c'è da dire che la GPU sembra attiva, ma non ci sono effetti notevoli sui tempi di inferenza. Ci sono due possibilità:
- L'attivazione è dovuta all'uso della GPU in fase di conversione del modello
- Anche se attivata in automatico (cosa che non credo) i tempi di inferenza sono comunque pessimi confrontati con Onnx, quindi non c'è un uso efficiente della GPU.
![[Schermata del 2025-02-15 15-55-25.png|Consumo Risorse TFLite con GPU Attiva]]


Test su Colab:
- Modello: ResNet152V2
- CPU Colab
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi
![[Schermata del 2025-02-15 15-38-48.png|Test Colab con CPU]]


# ONNX
Onnx offre due tipi di quantizzazione:
- Float16 (sperimentale)
- Int8

## Int8
Ci sono due casi:
- OperatorOriented
	- All the quantized operators have their own ONNX definitions, like QLinearConv, MatMulInteger and etc.
- TensorOriented
	- This format inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate the quantization and dequantization process.

Fasi:
1. Pre Elaborazione
2. Quantizzazione
3. Inferenza/Debugging

### Pre Elaborazione
Viene fatta un'inferenza sulle dimensioni!! Quindi Non è possibile avere modelli con size input variabile, ma solo fissa.

### Quantizzazione
Due casi:
- Dinamica
	- Lenta, meglio di no
- Statica
	- Serve dataset di calibrazione

Il vantaggio è la possibilità di quantizzare il modello per singoli livelli: La funzione di quantizzazione infatti è:

```python
def quantize_static(  
	model_input: str | Path | ModelProto,  
	model_output: str | Path,  
	calibration_data_reader: CalibrationDataReader,  
	quant_format: QuantFormat = QuantFormat.QDQ,  
	op_types_to_quantize: Any | None = None,  
	per_channel: bool = False,  
	reduce_range: bool = False,  
	activation_type: QuantType = QuantType.QInt8,  
	weight_type: QuantType = QuantType.QInt8,  
	nodes_to_quantize: Any | None = None,  
	nodes_to_exclude: Any | None = None,  
	use_external_data_format: bool = False,  
	calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,  
	extra_options: Any | None = None  
) -> None
```


> [!Quote] Documentazione di quantize_static
> Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file. It is recommended to use QuantFormat.QDQ format from 1.11 with activation_type = QuantType.QInt8 and weight_type = QuantType.QInt8. If model is targeted to GPU/TRT, symmetric activation and weight are required. If model is targeted to CPU, asymmetric activation and symmetric weight are recommended for balance of performance and accuracy.

In particolare:
- **op_types_to_quantize**. Specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
- **nodes_to_quantize**. List of nodes names to quantize. When this list is not None only the nodes in this list are quantized. example:
- **nodes_to_exclude**. List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.

Quindi una volta trovato il piano si potrebbe costruire un grafo a quantizzazione mista (in teoria... non provato nella pratica).

Test su Colab:
- Modello: ResNet152V2
- CPU Colab
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi

| Not Quantized      | ![[Schermata del 2025-02-14 17-15-14.png]] |
| ------------------ | ------------------------------------------ |
| **Int8 Quantized** | ![[Schermata del 2025-02-14 17-15-52.png]] |

In questo caso la quantizzazione sembra avere effetti più positivi anche senza GPU, a differenza di TFLite. Si avvalora quindi il fatto che TFLite abbia bisogno di un hardware apposito (o comunque su questo hardware di test non è ottimizzato).

Test su Istanza Cloud:
- Modello: ResNet152V2
- Macchina e2-standard-4
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi
![[Schermata del 2025-02-14 17-42-12.png|Run Onnx Su Istanza Cloud]]

In generale lo speed-up sta intorno all' 1.8.


Test su GPU:
- Modello: ResNet152V2
- GPU Google Colab
- Shape Input (1, 224, 224, 3)
- Numero di Run: 250
- Tempo in milli secondi
![[Schermata del 2025-02-15 15-00-43.png|Run Onnx Colab GPU]]
In questo caso la quantizzazione non apporta beneficio.


Sarebbe possibile anche eseguire un'ottimizzazione sul grafo a tempo di sessione, ma non sembra dare grande miglioramento rispetto al grafo quantizzato di base.

> [!NOTE] 
> Onnx offre anche la quantizzazione float16, in questo caso non testata perché è leggermente diversa rispetto a quella int8


# Onnx - Inference Size

Una volta quantizzato il modello in formato QDQ, quello che succede è quanto segue. Ogni operazione che supporta quantizzazione viene preceduta da dei livelli di dequantize e succeduta da dei livelli di quantize.

L'output di un layer di fatto viene calcolato come un Float32. Quando viene fatto passare per un livello di Quantize, viene quantizzato e convertito in Int8 (o comunque nel formato di quantizzazione usato).

Questa informazione può essere recuperata analizzando il value_info del grafo dopo l'operazione di infer_shape.
![[Schermata del 2025-02-15 19-18-14.png|Uno degli elementi di value_info]]

In questo caso abbiamo:
- Nome del tensore
- Il tensor type
	- Elem Type indica di che tipo di tensore si tratta
		- In questo caso 3 indica che si tratta di un tensore di int8
		- Il valore 1 invece indica un tensore di float32
		- Vedere tabella sotto

![[Schermata del 2025-02-15 19-20-50.png|Valori di elem_type|500]]
https://onnx.ai/onnx/api/mapping.html

Di conseguenza la quantità di dati da inviare è data da:
$tensor\_size * sizeof(tensor\_elem)$
E la tensor_size si può recuperare dagli stessi dati di cui sopra.

Nel nostro caso quindi, si può procedere così. L'ottimizzazione si può fare al netto degli operatori di quantizzazione/dequantizzazione. Una volta stabilito chi prende quali operatori e se questi operatori sono quantizzati si genera il modello e si distribuisce quello. Se un operatore è l'output di un sottomodello, la divisione si fa in maniera tale che ad essereinviato sia l'output quantizzato, e non quello da quantizzare, in modo da ridurre la dimensione dei dati inviati.

In alternativa, visto che gli operatori di quantize e dequantize usano capacità di calcolo, si può modellare questa cosa aggiungendo un termine alla latenza di calcolo: in questo modo si tiene anche in conto il consumo energetico dato dalla quantizzazione.

Un'altra cosa che si può fare è che, dato il modello, stabilito quali quantizzazioni usare, si possono mischiare nello stesso modello!! Si procede così:
1. Trovo i tipi di quantizzazione da usare per ogni livello
2. A questo punto partendo dal modello di base chiamo più volte la quantize_static usando il parametro di nodes_to_quantize per indicare quali devono essere quantizzati con la quantizzazione dell'iterazione corrente
3. Alla fine del ciclo ho un modello con quantizzazione mista che posso dividere con le funzioni onnx.

Un possibile problema potrebbe essere il calcolo dello speedup per singolo layer: lo speedup potrebbe dipendere da altri fattori complessivi e non da un unico layer (soprattutto potrebbe non dipendere da tutti i layer).


# Onnx - Speedup per layer

Lo speedup per layer si può calcolare abbastanza semplicemente usando le utils di Onnx per la divisione del modello. 
Tra i vari aspetti bisogna tenere in conto il fatto che il guadagno complessivo potrebbe non derivare da un unico layer, ma potrebbe derivare dall'interazione di più aspetti contemporaneamente.

Nell'immagine abbiamo (opName, opType) --> (normalTime, quantTime, speedUp, ...).
Come si vede dall'immagine ci sono dei livelli che danno speedup maggiore di uno e altri che invece lo hanno minore di uno.

Parametri Test:
- Architettura Locale
- NO GPU
- Input generati randomicamente
	- Dataset di Calibrazione
	- Dato di Test
![[Schermata del 2025-02-17 08-45-26.png|Speedup Per Layer]]

Credo comunque che l'aumento della velocità dato dalla modalità QDQ sia strano, visto il maggior numero di operazioni: il modello potrebbe comunque essere ottimizzato internamente per ridurre questa influenza.

Onnx permette anche di usare l'opzione QOperator mischiandola con il resto del modello in float32. L'esempio è in figura seguente: in questo caso l'operatore è quantizzato, ma le due operazioni prima e dopo non lo sono, quindi l'operatore quantizzato è circondato di operatori di quantize e dequantize prima e dopo che potrebbero essere dei punti di separazione e di invio dei dati.
![[Schermata del 2025-02-17 11-56-46.png]]

I modelli quantizzati con QOperators sono molto più lenti rispetto a quelli quantizzati con QDQ. La cosa è strana perché i QDQ dovrebbero introdurre un overhead maggiore rispetto ai QOperator, ma se l'hardware non è ottimizzato è possibile che ci siano dei rallentamenti nei QOperators (non sembrano esserci riferimenti a questo aspetto, in generale anche la documentazione di Onnx raccomanda l'uso di QDQ come si legge nella citazione precedente).


# Onnx - Errore Quantizzazione
Anche l'errore di quantizzazione non è difficile da valutare visto che si può applicare la quantizzazione per ogni layer e valutare il suo effetto sul totale. 

C'è da aggiungere che ci potrebbe essere una leggera perdita nella conversione dal modello originale ad Onnx, ma da alcuni test questa risulta abbastanza bassa.


# TFLite - Inference Size

Usando la libreria esterna *tflite* sembra possibile recuperare le informazioni riguardo la dimensione dei tensori di output per livello.
Anche in questo caso possiamo trovare il mapping tra codice del tipo e tipo di dato e fare lo stesso ragionamento di Onnx.

![[Schermata del 2025-02-17 13-58-08.png]]

Un possibile problema qui potrebbe essere l'integrazione tra livelli quantizzati in modi diversi: mentre Onnx permette di fare questa cosa, TFLite potrebbe avere dei problemi nella gestione di questo aspetto:
- Non sembra esserci una conversione automatica, quindi se il modello si aspetta di ricevere un certo tipo vuole quel tipo e la conversione deve essere fatta prima di dare l'input al modello stesso
- Non sembra essere possibile mischiare i tipi di quantizzazione. Mentre in Onnx si possono fare le valutazioni per poi quantizzare e poi dividere, in TFLite non c'è la possibilità di specificare un sottoinsieme di livelli da quantizzare, quindi o tutti i livelli del sottomodello vengono quantizzati o nessuno lo è. Questo riduce la flessibilità

Di fatto il problema delle quantizzazioni diverse potrebbe porsi solo con la quantizzazione Full Integer, in cui sono impostati i tipi che il modello vuole in input e in output. Con gli altri tipi di quantizzazione il problema non dovrebbe sorgere visto che anche le prove precedenti hanno mostrato la possibilità di interagire.

# TFLite - Speedup per Layer
Al netto del fatto che la quantizzazione in TFLite non sembra apportare grandi benefici durante l'esecuzione, lo speedup per layer si può calcolare facendo i seguenti passi:
1. Si divide il modello layer per layer (partendo dalla divisione del modello keras)
2. Il sottolivello viene esportato come modello TFLite quantizzato e non quantizzato
3. Si confrontano i risultati dei tempi

(Non ancora fatta una prova).

Un'alternativa potrebbe essere dividere il modello già convertito, ma bisogna eventualmente trovare dei tool che permettano di farlo.

# TFLite - Errore di Quantizzazione
Si pone qui lo stesso problema dell'inferenza relativo alla possibile interazione di quantizzazioni diverse.

Oltre a questo, può risultare difficile valutare l'effetto della quantizzazione per singolo layer o per gruppo di layer proprio perché non c'è modo di quantizzare una singola parte del modello. Per fare questo bisognerebbe prima divider il modello, quantizzare la parte voluta e poi fare l'inferenza in due fasi.

In questo senso un'alternativa potrebbe essere quella riportata in:
- https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/quantization/keras/quantize_apply
- https://github.com/tensorflow/tensorflow/issues/45887
L'approccio qui riportato sembra essere simile a quello di Onnx, ma bisogna capire bene se ciò che è dato qui vale in fase di training o anche in fase di inferenza. Da quello che sembra vengono aggiunti dei livelli di quantize e dequantize allo stesso modo di Onnx che potrebbero (??) dare lo stesso comportamento.



# Onnx - Test Combinazioni
Sono stati combinati i vari formati e tipi di quantizzazioni di Onnx.

I test seguenti sono stati fatti con:
- ResNet112
- Input (1, 3, 224, 224)
- Numero di Run = 100
Il Test seguente è stato fatto con:
- Macchina e2-standard-4 (no support VNNI)
- Sistema operativo `debian-cloud/debian-11`
![[Schermata del 2025-02-25 10-14-05.png|Test Quantizzazioni - e2 Standard 4]]
Come si vede dal test i casi che restituiscono i risultati migliori sono quelli in cui le attivazioni sono quantizzate come unsigned int e i pesi come signed.

Il test seguente è stato fatto con:
- Macchina `c3-standard-4` (Supporto VNNI - https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c3_series)
- Sistema Operativo `debian-cloud/debian-11`
![[Schermata del 2025-02-25 10-34-31.png| Test Quantizzazioni - c3 Standard 4]]
Come si vede dal test in generale la computazione risulta più veloce rispetto al test corrispondente fatto sull'altra macchina.

Notare che le prestazioni della combinazione (QOperator, QInt8, QInt8), sono pessime in entrambi i casi; il warning stesso dice che questa combinazione non è adatta ad architetture x86.


Il test seguente è stato fatto con:
- Macchina c3d-standard-4 (https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c3d_series)
- Sistema operativo `debian-cloud/debian-11`
![[Schermata del 2025-02-25 10-57-11.png|Test Quantizzazioni - c3d Standard 4]]
Anche in questo caso, trovandoci comunque su architetture x86_64, le performance del caso (QOperator, QInt8, QInt8) sono comunque basse, anche se migliori rispetto agli altri due casi (possibili fluttuazioni??).


Il test seguente è stato fatto con:
- Macchina `c4a-standard-4` (https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c4a_machine_types)
- Sistema Operativo `debian-cloud/debian-12-arm64`
![[Schermata del 2025-02-25 11-22-39.png|Test Quantizzazioni - c4a-standard-4]]
Finalmente su architettura ARM la combinazione peggiore mostra buone performance, confrontabili con quella degli altri casi.



> [!Quote] Risposta da ChatGPT:
>The main reason why **S8S8 mode is slower in QOperator** on x86-64 CPUs is due to the way integer matrix multiplications (e.g., **INT8 GEMM/convolutions**) are handled in different execution paths within ONNX Runtime. Here’s a breakdown:
>### 1. **Lack of Efficient INT8 Computation in QOperator Path**
>- ONNX Runtime supports two execution paths for quantized models:
    >- **QDQ (Quantize-Dequantize)**: Uses quantized tensors but leverages optimized computation kernels (e.g., **Intel MKL-DNN (oneDNN)**, QNNPACK, etc.).
    >- **QOperator (Quantized Operators)**: Directly executes pre-quantized operators in the ONNX graph.
    >- The problem with **S8S8 in QOperator mode** is that many x86-64 CPUs (especially those without AVX512-VNNI or AMX instructions) **lack optimized INT8×INT8 (S8S8) computation**. Most optimized implementations (like those in oneDNN) **favor U8S8** (unsigned 8-bit × signed 8-bit) or **U8U8** for better vectorization.
> ### 2. **AVX2 and AVX-512 Limitation**
> - Many CPUs rely on **AVX2** for integer matrix multiplication, but AVX2 does not natively support **S8S8 multiplication efficiently**.
> - On AVX-512 CPUs, **AVX512-VNNI** can accelerate U8S8 computations, but efficient **S8S8 support requires newer AMX instructions (Intel Sapphire Rapids and later)**.
> - If the CPU does not support these advanced instructions, **ONNX Runtime must fall back to slower implementations**, like naive loops or scalar multiplications.
> ### 3. **Data Zero-Point Adjustments in QOperator**
> - **U8S8 is often more optimized** because:
    >- Many hardware accelerations assume unsigned activation (U8) and signed weights (S8), as this pattern is common in low-precision ML workloads.
    >- **S8S8 requires extra handling for zero-points**, making it **less efficient in QOperator mode** due to additional preprocessing overhead.
> ### 4. **QDQ Mode Optimizations**
> - In contrast, **S8S8 in QDQ mode is well-optimized** for performance because it allows ONNX Runtime to **leverage fused kernels** from libraries like **oneDNN**.
> - The **QDQ approach balances accuracy and speed** because it inserts quantization nodes dynamically, allowing the runtime to apply the best execution strategy.
> ### **Conclusion**
> 
**S8S8 mode in QOperator is slow because x86-64 CPUs do not efficiently support INT8×INT8 multiplication, requiring inefficient fallback implementations. Instead, U8S8 or U8U8 are often faster due to better hardware support.** If performance is critical, **S8S8 should be used with QDQ instead of QOperator** to leverage optimized execution paths.


Dal sitop di google cloud (https://cloud.google.com/compute/docs/cpu-platforms?hl=it#requirements_for_using_amx):

> [!Quote] https://cloud.google.com/compute/docs/cpu-platforms?hl=it#requirements_for_using_amx
> Compute Engine offre il supporto per AMX nelle seguenti [immagini pubbliche](https://cloud.google.com/compute/docs/images/os-details?hl=it):
> - CentOS Stream 8 o versioni successive
> - Container-Optimized OS 109 LTS o versioni successive
> - RHEL 8 (build più recente) o versioni successive
> - Rocky Linux 8 (build più recente) o versioni successive
> - Ubuntu 22.04 o versioni successive
> - Windows Server 2022 o versioni successive

Inoltre viene detto che:

> [!Quote] https://cloud.google.com/compute/docs/cpu-platforms?hl=it#intel-amx
> AMX è supportato sui processori Intel Xeon di 5ª generazione (nome in codice Emerald Rapids), che alimentano la serie di [VM per uso generico C4](https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c4_series), nonché sui processori Intel Xeon di 4ª generazione (nome in codice Sapphire Rapids), che alimentano le VM A3 ottimizzate per l'acceleratore e le [VM per uso generico C3](https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c3_series). Tutti i tipi di macchine VM C4 e C3 supportano gli insiemi di istruzioni AMX.


Quindi la versione debian usata finora non supporta AMX; proviamo una versione ubuntu!!

Il seguente test è stato fatto su:
- Macchina `c3-standard-4` (Supporto VNNI - https://cloud.google.com/compute/docs/general-purpose-machines?hl=it#c3_series)
- Sistema Operativo `ubuntu-pro-2404-lts-amd64`
![[Schermata del 2025-02-25 11-53-03.png|Test Quantizzazioni - c3 standard 4 con Ubuntu]]
La cosa non sembra apportare beneficio; a questo punto credo sia proprio un solo fattore architetturale che impedisce un uso efficiente.


A questo punto quindi a meno di non usare ARM, la quantizzazione (QOperator, QInt8, QInt8) credo sia fuori questione visto lo slowdown che apporta al modello.
Tutte le altre quantizzazioni sembrano invece fattibili, anche se alcune danno uno speedup maggiore di altre.

> [!Warning] 
> La quantizzazione è stata testata con dataset di calibrazione e con degli input dummy, quindi i tempi ottenuti in situazioni reali potrebbero essere diversi da quelli trovati in questo caso.


# Onnx - Test Quantizzazione Mista

Con quantizzazione mista intendo usare la quantizzazione (QDQ, Int8, Int8) per tutti i livelli che vengono quantizzati: misto si riferisce al fatto che non tutti lo sono. In particolare si fa prima un'estrazione dei singoli livelli, li si quantizza e si vede quali sono i livelli per cui la quantizzazione apporta beneficio. Dopo viene costruito un solo modello in cui si quantizzano solo i livelli con speedup > 1.

Parametri di Test:
- Modello ResNet50V2
- Run
	- 10 per Calcolo Speedup del singolo livello 
	- 150 per Calcolo Tempi dei modelli complessivi
- Architettura: Mio PC (Intel i7 8th)
![[Schermata del 2025-02-25 16-41-03.png|Confronto Tempi Esecuzione]]

Come si vede dallo screen i tempi sono migliori per la quantizzazione mista, anche se non di moltissimo.


# Onnx - Test Ottimizzazioni di Quantizzazione

Test fatto su:
- Modello Yolo11n-Segmentation
- Mio PC

Nota: 
- QDQ --> Aggiunta di Livelli intermedi per la gestione della quantizzazione (simile a quello che si fa per gli addestramenti basati su quantizzazione)
- QOperator --> Uso di operatori appositi per la modellazione delle operazioni quantizzate

![[Schermata del 2025-03-03 10-26-58.png|QDQ, UINT8, BASIC_OPT]]

| ![[Schermata del 2025-03-03 10-28-16.png\|QDQ, BASIC_OPT, Grafo Quantizzato Ottimizzato\|300]] | ![[Schermata del 2025-03-03 10-29-29.png\|BASIC_OPT, Grafo Non Quantizzato Ottimizzato\|300]] |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |


![[Schermata del 2025-03-03 10-31-42.png|QOperator, UINT8, BASIC_OPT]]

![[Schermata del 2025-03-03 10-32-39.png|QOperator, BASIC_OPT, Grafo Quantizzato Ottimizzato|300]]



![[Schermata del 2025-03-03 10-34-54.png|QDQ, UINT8, EXTENDED_OPT]]


| ![[Schermata del 2025-03-03 10-35-47.png\|EXTENDED_OPT, Not Quantized\|300]] | ![[Schermata del 2025-03-03 10-36-33.png\|QDQ, EXTENDED_OPT, Quantized\|300]] |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |


![[Schermata del 2025-03-03 10-39-15.png|QOperator, EXTENDED_OPT]]

![[Schermata del 2025-03-03 10-49-00.png|QDQ, ALL_OPT]]

![[Schermata del 2025-03-03 10-50-25.png|QOperator, ALL_OPT]]


Questi esperimenti mostrano quindi come i QOperator e la QDQ siano in sostanza equivalenti in termini di calcolo nel momento in cui vengono eseguite tutte le ottimizzazioni del caso (sarebbe stato fin troppo strano il contrario).

Il problema dei QOperators è che sono degli operatori implementati in onnxruntime che è della cara vecchia Microsoft (Microsoft regala sempre grandi soddisfazioni...). Questi operatori non sono supportati dal profiler che ho usato per i FLOPS e dagli strumenti di partizionamento del modello di Onnx. I livelli QuantizeLinear e DequantizeLinear sono invece supportati perché definiti proprio in Onnx (non nel runtime).

La cosa migliore per lavorare sulla rete quantizzata potrebbe quindi essere quella di lavorare su una rappresentazione QDQ. Una volta trovato lo splitting in questo formato si fa la divisione e da lì sarà poi la fase di inferenza ad ottimizzare ed eventualmente ad unire gli operatori aggregandoli. In questo senso ci sono alcuni aspetti da considerare:
- Come considerare i FLOPS dei livelli di Quantize e Dequantize?
	- Si potrebbero escludere dal calcolo complessivo e assumere già che saranno incorporati nell'unico QOperator
		- Di fatto questo non è proprio sempre verissimo, perché ci sono alcuni operatori che fanno proprio quantize e dequantize internamente oppure fanno altre operazioni, come ad esempio
			- https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqlinearadd
			- https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqlinearsigmoid
	- Si potrebbero considerare, ma questo poi a livello di ottimizzazione potrebbe creare dei paradossi per cui il modello quantizzato ha un costo maggiore di quello non quantizzato
	- Si potrebbero considerare ma scontandoli in qualche modo, considerando quindi un valore più basso di quello dato dal profiler
- Si dovrebbe imporre che i livelli di quantize e di dequantize che sono usati per la modellazione del livello come QDQ non siano separabili, ma siano obbligatoriamente mappati nello stesso nodo, perché altrimenti l'ottimizzatore non potrebbe riunirli nel QOperator corrispondente (non dovrebbe essere complicato, basterebbe aggiungere un attributo agli archi per indicare se è mappabile su un arco della rete o meno)

(GRAZIE ANCORA A MICROSOFT, SEMPRE AL TOP)

Da questi esperimenti non sembra che ci sia uno speedup significativo; in realtà questo è dovuto in buona parte alla dimensione del modello. Aumentando la dimensione e la complessità diventa molto più evidente
Parametri Test:
- Macchina c4-standard-4
- Numero di Run 200
- Yolo11x-Segmentation (versione più grande possibile per modelli Yolo)
- Quantizzazione con QOperator con ottimizzazione default (massima) per il runtime
- Tempi in milli secondi
![[Schermata del 2025-03-03 12-09-24.png]]
Per modelli molto grandi lo speedup cresce molto, qui siamo sui 3.75, passando da circa un secondo per modelli non quantizzati ad circa 300 ms per modelli quantizzati.

Il test successivo è stato fatto con modello QDQ sempre per Yolo11x-seg sulla stessa macchina: anche qui si può vedere come le prestazioni siano praticamente le stesse rispetto al modello costruito direttamente con QOperator.
![[Schermata del 2025-03-03 12-21-06.png]]

Prova definitiva: eseguendo il profiling con l'opzione della InferenceSession, si può vedere come gli operatori che sono eseguiti sono proprio dei QOperators.
![[Schermata del 2025-03-04 09-16-10.png]]