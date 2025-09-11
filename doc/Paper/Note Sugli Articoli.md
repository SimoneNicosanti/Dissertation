Legenda Colori:
- Blu, aspetti di rilevanza generale
- Rosa, aspetti specifici dello studio
- Giallo, aspetti sperimentali

# CoEdge

## Introduzione

> [!Quote|] Abstract [[CoEdge.pdf#page=1&selection=34,0,47,8|CoEdge, p.1]]
> > Abstract— Recent advances in artificial intelligence have driven increasing intelligent applications at the network edge, such as smart home, smart factory, and smart city. To deploy computationally intensive Deep Neural Networks (DNNs) on resourceconstrained edge devices, traditional approaches have relied on either offloading workload to the remote cloud or optimizing computation at the end device locally. However, the cloud-assisted approaches suffer from the unreliable and delay-significant widearea network, and the local computing approaches are limited by the constrained computing capability. Towards high-performance edge intelligence, the cooperative execution mechanism offers a new paradigm, which has attracted growing research interest recently
> 
> 


> [!PDF|128, 235, 255] [[CoEdge.pdf#page=1&annotation=265R|CoEdge, p.1]]
> > Motivated by this success, it is envisioned that employing DNNs to edge devices would enable and boost employing DNNs to edge devices would enable and boost intelligent services, supporting brand new smart interactions intelligent services, supporting brand new smart interactions between humans and their physical surroundings.
> 
> 

> [!PDF|128, 235, 255] [[CoEdge.pdf#page=1&annotation=266R|CoEdge, p.1]]
> > Therefore, minimizing response latency and promising users’ experience is of paramount importance.
> 
> 


> [!PDF|note] [[CoEdge.pdf#page=1&selection=230,1,234,24&color=note|CoEdge, p.1]]
> > What’s worse, for many smart applications with human in the loop, the sensory data can contain highly sensitive or private information. Offloading these data to the remote datacenter owned by curious commercial companies inevitably raises users’ privacy concerns.

Potrei citare l'articolo insieme a questo (per quanto non so bene di cosa tratti).
> [!PDF|note] [[CoEdge.pdf#page=2&selection=23,30,27,49&color=note|CoEdge, p.2]]
> > edge intelligence paradigm [13]. Instead of uploading data to the remote cloud or keeping all computation at the single local device, edge intelligence enjoys real-time response as well as privacy preservation by offloading computing workload within a manageable range.

> [!PDF|note] [[CoEdge.pdf#page=2&selection=45,55,51,7&color=note|CoEdge, p.2]]
> > Key challenges to be addressed: (1) how to decide the workload assignment tailored to the resource heterogeneity of edge devices, (2) how to optimize the system performance with the presence of network dynamics, and (3) how to orchestrate computation and communication during cooperative inference runtime

> [!PDF|important] [[CoEdge.pdf#page=2&selection=63,0,82,7&color=important|CoEdge, p.2]]
> >  CoEdge (Cooperative Edge), a runtime system that orchestrates cooperative DNN inference over multiple heterogeneous edge devices
> 
> 

> [!PDF|important] [[CoEdge.pdf#page=2&selection=138,50,181,0&color=important|CoEdge, p.2]]
> >  CoEdge optimally partitions the input inference workload, where the partitions’ sizes are chosen to match devices’ computing capabilities and network conditions to improve system performance in both latency and energy metrics.


## Motivazioni
> [!PDF|note] [[CoEdge.pdf#page=2&selection=311,1,313,58&color=note|CoEdge, p.2]]
> > The key impediment of deploying CNN at the network edge lies in the gap between intensive CNN inference computation and the limited computing capability of edge devices. , we can utilize the cooperative inference mechanism to exploit available computing resources at the edge.
> 
> 

> [!PDF|note] [[CoEdge.pdf#page=4&selection=28,1,39,10&color=note|CoEdge, p.4]]
> > To effectively exploit computing resources at the edge, we need to felicitously factor the computing capabilities of edge devices considering magnitude and heterogeneity
> 
> > More specifically, it is crucial to decide which device to participate in the cooperative inference and how much workload each device affords
> 


## Design and Workflow

### Design
> [!PDF|important] [[CoEdge.pdf#page=4&selection=74,33,83,1&color=important|CoEdge, p.4]]
> > For the device that launches a CNN inference task, we label it as the master device, and for the device that joins the cooperation, it is marked as the worker device.

> [!PDF|important] [[CoEdge.pdf#page=4&selection=101,20,126,14&color=important|CoEdge, p.4]]
> > In the setup phase, CoEdge records the execution profiles of each device. In the runtime phase, CoEdge creates a cooperative inference plan that determines the workload partitions and their corresponding assignment, using the profiling results collected in the setup phase and the network status
> 
> 

Nella fase di setup l'intero modello viene fatto eseguire ai vari dispositivi partecipanti al sistema per trovare le varie capacità di esecuzione dei dispositivi! La cosa credo dipenda dal tipo di modello: 
- se il modello è medio piccolo può anche andare, ma se non lo è non so quanto possa funzionare la cosa; 
- dipende molto anche dalla dimensione del modello: se il modello è troppo grande per entrare in memoria potrebbe non essere proprio possibile l'esecuzione (per quanto in realtà potrebbe entrare in gioco lo swap a quel punto). 
Si tratterebbe comunque di un costo pagato una sola volta in fase di registrazione del server nel sistema: questo potrebbe permettere anche aiutare per la parte di profilazione della quantizzazione.
> [!PDF|important] [[CoEdge.pdf#page=4&selection=147,1,179,25&color=important|CoEdge, p.4]]
> > **Setup phase**. Whenever a CNN-based application is installed, Device Profiler runs the CNN models locally and records Profiling Results. These results sketch the device’s computing capability, including the computation intensity, computation frequency and power parameters, which will be detailed in Section IV-A.
> 
> 

### Workflow
> [!PDF|important] [[CoEdge.pdf#page=4&selection=205,15,212,48&color=important|CoEdge, p.4]]
> > We exploit model parallelism to partition CNN inference over multiple devices. Under model parallelism, CNN model parameters are divided into subsets and assigned to multiple edge nodes. With respective parameters, each device accepts a necessary part of the input feature maps and generates a portion of the output feature maps. 

> [!PDF|important] [[CoEdge.pdf#page=4&selection=217,39,219,21&color=important|CoEdge, p.4]]
> > To accommodate devices’ heterogeneity, the partition sizes are differentiated to match device capabilities. 
> 
> 

## Partitioning
> [!PDF|note] [[CoEdge.pdf#page=6&selection=51,1,63,53&color=note|CoEdge, p.6]]
> > We assume that the devices are available and relatively stable during the inference runtime. This can be relevant as executing an inference task is typically in a period of seconds, and many edge environments are maintained statically in independent spaces, such as smart home and smart factory. Besides, the underlying support of intelligent services in such scenarios usually employs a few commonly-adopted DNN models and frequently run similar types of DNN inference tasks. Therefore, we suppose that the DNN models have been loaded ahead of inference queries, and can be used to compute input tensors as soon as necessary data are prepared.
> 
> 


Controllare il paper 26: può avere delle considerazioni interessanti sull'aspetto energetico
> [!PDF|note] [[CoEdge.pdf#page=6&selection=268,1,322,38&color=note|CoEdge, p.6]]
> >  A resource tuple (ρ, f, m, P c, P x)i specifics the resource profile of device i. Here, ρ is defined as the computing intensity (in processing cycles per 1KB input) of the given DNN model, which is measured by applicationdriven offline profiling [[Energy_Efficiency_mobile_clients_cloud_computing.pdf|26]] in the setup phase. f is the device’s CPU frequency, reflecting its computing capability in a coarse granularity. m is the available maximum memory capacity for inference tasks. For a single device that only processes CNN workloads, m is the volume of memory excluding the space taken by the underlying system services, e.g., I/O services, compiler, etc. P c and P x denote the computation power and the wireless transmission power, respectively.
> 
> 


Il problema dall'assumere questa proporzionalità è che sicuramente dipende dal tipo di aritmetica e dall'hardware che si considera!
> [!PDF|note] [[CoEdge.pdf#page=6&selection=459,1,466,61&color=note|CoEdge, p.6]]
> > During a single layer’s execution, the system takes time and energy on two aspects, computation and communication. For computation, we calculate the latency and energy by first approximating the computing cycles of given partitions. As demonstrated in previous empirical studies [26], [28], [29], for many data processing tasks as exemplified by data encoding and decoding, the required computing cycles are proportional to their input data sizes. This means that, a constant computing intensity (in computing cycles per unit data) exists for such tasks, and we can use it to capture the effective computing capability of a specific device. Existing literature, such as [30]–[32], has leveraged this observation to characterize deep learning workloads, and in this work, we adopt it to estimate the computing cycles amount given the partitions and DNN layers
> 
> 
> 
> 


Questo è un altro aspetto importante: in questo caso si sta tenendo in considerazione il fatto che i vari dispositivi stanno lavorando in parallelo, perché l'input viene diviso in modo tale che tutto il modello sia calcolato su questo input. **Noi il parallelismo non lo stiamo considerando nella risoluzione del problema.**
> [!PDF|important] [[CoEdge.pdf#page=7&selection=293,1,293,61&color=important|CoEdge, p.7]]
> > During the interval between two synchronizations, there is no data dependency between devices, and thus they process jobs in parallel. These scattered feature map partitions are finally aggregated at the classification stage for FC computation. Hence, the whole process works in a Bulk Synchronous Parallel (BSP) mechanism [34].
> 
> 

## Esperimenti
> [!PDF|yellow] [[CoEdge.pdf#page=2&selection=183,35,228,6&color=yellow|CoEdge, p.2]]
> > prototype with Raspberry Pi 3, Jetson TX2, and desktop PC. Experimental evaluations show 7.21× ∼ 4.49× latency speedup over the local approach and up to 25.5% ∼ 66.9% energy saving

> [!PDF|yellow] [[CoEdge.pdf#page=3&selection=74,1,95,14&color=yellow|CoEdge, p.3]]
> > We measure the end-to-end latency of this process, i.e., from the image input to the inference result output; and we record the average latency of fulfilling the inference task over 100 runs.
> 
> 

> [!PDF|yellow] [[CoEdge.pdf#page=3&selection=97,26,98,62&color=yellow|CoEdge, p.3]]
> > For the bandwidth between two devices, we fix it at 1MB/s using the traffic control tool tc [2

> [!PDF|] [[CoEdge.pdf#page=9&selection=40,13,44,50|CoEdge, p.9]]
> >  For each CNN model, we run it for once as warm-up and then record the execution time with 50 runs without break. The aim of warm-up running is to alleviate the impact of weight loading and TensorFlow initiation since we have omitted these overheads in the formulation
> 
> 


### Scalability
> [!PDF|yellow] [[CoEdge.pdf#page=11&selection=28,1,30,11&color=yellow|CoEdge, p.11]]
> > To evaluate CoEdge’s scalability, we measure the latency and energy by incrementally adding devices to the experimental cluster
> 
> 

> [!PDF|yellow] [[CoEdge.pdf#page=11&selection=36,8,53,1&color=yellow|CoEdge, p.11]]
> >  With the increase of the cluster scale, both the latency and dynamic energy drop. In particular, there is a distinctive decrease when adding PC (2 → 3) and Jetson TX2 (5 → 6).



## Specifiche energetiche dei dispositivi usati
Di seguito le specifiche hardware dei dispositivi usati in CoEdge.

> [!PDF|yellow] [[CoEdge.pdf#page=3&selection=68,51,70,19&color=yellow|CoEdge, p.3]]
> > Which are measured with Monsoon High Voltage Monitor [17] using the methodology in [18]
> 

Volendo si potrebbero assumere questi valori prendendoli per buoni e citando il paper, anche se in questo caso non si sta considerando il continuum, ma solo l'edge.

![[CoEdge.pdf#page=3&rect=71,362,277,610&color=note|CoEdge, p.3|400]]
![[CoEdge.pdf#page=9&rect=323,623,553,740&color=note|CoEdge, p.9|400]]



# [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=2&selection=0,0,0,53|Energy_Efficiency_mobile_clients_cloud_computing]]

> [!PDF|note] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=2&selection=63,0,65,50&color=note|Energy_Efficiency_mobile_clients_cloud_computing, p.2]]
> > In our analysis, we discuss the computing to communication ratio, which is the critical factor for the decision between local processing and computation offloading

> [!PDF|note] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=2&selection=68,9,72,0&color=note|Energy_Efficiency_mobile_clients_cloud_computing, p.2]]
> > Additionally, not only the amount of transferred data but also the traffic pattern is important; sending a sequence of small packets consumes more energy than sending the same data in a single burst.



## Background

> [!PDF|note] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=3&selection=4,26,8,33&color=note|Energy_Efficiency_mobile_clients_cloud_computing, p.3]]
> > For mainstream cloud computing the most important concern is the time and the cost of transferring massive amounts of data to the cloud while for mobile cloud computing the key issue is the energy consumption of the communication.

## Trade-off Analysis

In questo caso E_cloud è considerata l'energia di spostare la computazione su cloud, non quella per eseguire la computazione sul cloud: per questo motivo in locale considera solo il consumo dato dal calcolo e per il cloud considera solo il consumo dato dalla trasmissione.
> [!PDF|note] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=3&selection=93,35,107,2&color=note|Energy_Efficiency_mobile_clients_cloud_computing, p.3]]
> > the critical aspect for mobile clients is the trade-off between energy consumed by computation and the energy consumed by communication. We need to consider the energy cost of performing the computation locally (E_local) versus the cost of transferring the computation input and output data (E_cloud).

![[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=3&rect=311,229,546,329&color=note|Energy_Efficiency_mobile_clients_cloud_computing, p.3]]

> [!PDF|yellow] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=4&selection=94,28,100,27&color=yellow|Energy_Efficiency_mobile_clients_cloud_computing, p.4]]
> > Figure 1 illustrates the dependency of energy per transferred data on communication bit-rate. As can be seen, the higher the bit-rate, the more energy efficient the data transfer is. The figure also illustrates the fact that the energy efficiency of cellular communication tends to be more sensitive to the data transfer bit-rate than WLAN

![[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=4&rect=317,546,549,713&color=yellow|Energy_Efficiency_mobile_clients_cloud_computing, p.4]]

> [!PDF|yellow] [[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=4&selection=108,0,113,26&color=yellow|Energy_Efficiency_mobile_clients_cloud_computing, p.4]]
> > Table 2 lists the energy characteristics of wireless communication for the Nokia N810 and N900, measured with the netperf TCP streaming benchmark. The WLAN throughput of the N810 is affected significantly by the CPU operating point so the metrics are shown for all operating points separately.

![[Energy_Efficiency_mobile_clients_cloud_computing.pdf#page=4&rect=311,90,575,251&color=yellow|Energy_Efficiency_mobile_clients_cloud_computing, p.4]]


# [[energy_optimal_mobile_application_execution.pdf#page=1&selection=0,0,0,44|energy_optimal_mobile_application_execution]]

> [!PDF|note] [[energy_optimal_mobile_application_execution.pdf#page=1&selection=86,0,94,61&color=note|energy_optimal_mobile_application_execution, p.1]]
> In this paper, we focus on the problem of energy-optimal application execution in the cloud-assisted mobile platform. The objective is to minimize the total energy consumed by the mobile device. When the applications are executed in the mobile device, the computation energy can be minimized by optimally scheduling the clock frequency of the mobile device. When the applications are executed in the cloud clone, the transmission energy can be minimized by optimally scheduling the transmission data rate via a stochastic wireless channel.


> [!PDF|yellow] [[energy_optimal_mobile_application_execution.pdf#page=1&selection=101,23,105,60&color=yellow|energy_optimal_mobile_application_execution, p.1]]
> Our numerical results indicate that the optimal policy depends on the application profile (i.e., the input data size and the delay deadline) and the wirelesstransmission model. Moreover, the cloud execution can result in significant amount of energy saving for the mobile device.



> [!PDF|important] [[energy_optimal_mobile_application_execution.pdf#page=2&selection=24,0,56,49&color=important|energy_optimal_mobile_application_execution, p.2]]
> We denote an application profile as A(L, T ), where L and T are the two parameters for the application given as follows: • Input data size L: the number of data bits as the input to the application; • Application completion deadline T : the delay deadline before which the application should be completed.


## Mobile Execution Model
> [!PDF|note] [[energy_optimal_mobile_application_execution.pdf#page=2&selection=69,0,115,25&color=note|energy_optimal_mobile_application_execution, p.2]]
> For the mobile execution, its computation energy can be minimized by optimally configuring the clock frequency of the chip, via the dynamic voltage scaling (DVS) technology [11]. In CMOS circuits [3], the energy per operation Eop is proportional to V 2, where V is the supply voltage to the chip. Moreover, it has been observed that the clock frequency of the chip, f , is approximately linearly proportional to the voltage supply of V [3]. Therefore, the energy per operation can be expressed as Eop = κf 2, where κ is the energy coefficient depending on the chip architecture. The optimization problem can then be formulated as

![[energy_optimal_mobile_application_execution.pdf#page=2&rect=47,77,304,150&color=note|energy_optimal_mobile_application_execution, p.2]]

## Cloud Execution Model
> [!PDF|important] [[energy_optimal_mobile_application_execution.pdf#page=2&selection=163,32,175,6&color=important|energy_optimal_mobile_application_execution, p.2]]
> For any mobile application A(L, T ), L bits of data needs to be transmitted to the cloud clone.

> [!PDF|important] [[energy_optimal_mobile_application_execution.pdf#page=2&selection=177,30,182,33&color=important|energy_optimal_mobile_application_execution, p.2]]
> We assume a Markovian fading model for the wireless channel between the mobile device and the cloud clone. A specific model (i.e., the Gilbert-Elliott model) for the channel gain will be presented in Section IV-A. In this research, we adopt an empirical transmission energy model as in [7], [8] [18], [19]. 

![[energy_optimal_mobile_application_execution.pdf#page=2&rect=309,433,570,575&color=important|energy_optimal_mobile_application_execution, p.2]]

## Optimality
> [!PDF|note] [[energy_optimal_mobile_application_execution.pdf#page=2&selection=281,0,283,47&color=note|energy_optimal_mobile_application_execution, p.2]]
> The decision for optimal application execution is to choose where to execute the application, with an objective to minimize the total energy consumed by the mobile device.



> [!TODO] Vedere la parte di modellazione del canale di comunicazione



# [[Furcifer_PerCom24.pdf#page=1&selection=0,0,0,8|Furcifer]]

> [!Quote] Abstract [[Furcifer_PerCom24.pdf#page=1&selection=30,1,57,43|Furcifer_PerCom24, p.1]]
> Modern real-time applications widely embed compute intense neural algorithms at their core. Current solutions to support such algorithms either deploy highly-optimized Deep Neural Networks at mobile devices or offload the execution of possibly larger higher-performance neural models to edge servers. While the former solution typically maps to higher energy consumption and lower performance, the latter necessitates the low-latency wireless transfer of high volumes of data. Timevarying variables describing the state of these systems, such as connection quality and system load, determine the optimality of the different computing configurations in terms of energy consumption, task performance, and latency.

## Introduzione
> [!PDF|note] [[Furcifer_PerCom24.pdf#page=1&selection=115,35,125,44&color=note|Furcifer_PerCom24, p.1]]
> Challenges include constrained computing capabilities and energy budget of mobile devices, as well as communication channel capacity. Local Computing (LC) and Edge Computing (EC) stand as the primary strategies for tackling the broad range of real-world heterogeneous tasks centered on the execution of complex data analysis and decision making algorithms

> [!PDF|note] [[Furcifer_PerCom24.pdf#page=1&selection=170,0,191,27&color=note|Furcifer_PerCom24, p.1]]
> Recently, a third paradigm - Split Computing (SC), where sections of ML models optimized to facilitate offloading are allocated to the mobile device and edge server - emerged as a promising alternative to EC and LC (see Figure 1). Indeed, the most advanced SC frameworks, where specialized models embed neural encoder/decoder-like structures, result in minimal computing load to the mobile device, while considerably reducing network usage [1].

![[Furcifer_PerCom24.pdf#page=1&rect=315,389,564,546&color=note|Furcifer_PerCom24, p.1]]

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=2&selection=40,58,65,0&color=important|Furcifer_PerCom24, p.2]]
> Furcifer transparently monitors the state of the underlying system, evaluating at runtime the feasibility of EC, LC, and SC configurations in highly dynamic environments, and switch between them. The core of Furcifer is a new containerized approach that can effectively support the dynamic transition between EC, LC and SC



## Related Works
> [!PDF|note] [[Furcifer_PerCom24.pdf#page=2&selection=249,17,259,49&color=note|Furcifer_PerCom24, p.2]]
> These evaluations often overlook the significant performance degradation caused by image compression [15], [16], which is inevitable in practical EC systems. In particular, widespread image compression techniques are designed for human perception rather than for image analysis. As a consequence, high performance requires the transfer of large volumes of data over capacity-constrained channel

> [!PDF|note] [[Furcifer_PerCom24.pdf#page=2&selection=262,1,276,11&color=note|Furcifer_PerCom24, p.2]]
> SC (also known as supervised compression in some contexts) [17], [18] has recently emerged as a promising alternative to achieve state-of-the-art performance in computer vision tasks while effectively reducing bandwidth usage. The idea is to incorporate encoder/decoder-like structures within the ML models themselves, and use specialized training techniques to train task-oriented compressed representations [19], [20]. Knowledge Distillation [21], [22] is one of the tools used to maximize the effectiveness of SC frameworks.

> [!PDF|note] [[Furcifer_PerCom24.pdf#page=3&selection=44,1,51,37&color=note|Furcifer_PerCom24, p.3]]
> n the domain of real-time computer vision, energy consumption is not solely determined by the number of Floating Point Operations (FLOPs) or Multiply-Accumulate (MAC) operations indicative of the model’s complexity. Indeed, energy consumption is also proportional to the number of frame per seconds (F P S) processed by the system [27], [28].

> [!Quote] Understand the Definitions
> - A FLOP (Floating Point OPeration) is considered to be either an addition, subtraction, multiplication, or division operation.
>- A MAC (Multiply-ACCumulate) operation is essentially a multiplication followed by an addition, i.e., MAC = a * b + c. It counts as two FLOPs (one for multiplication and one for addition).
>
>https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc


> [!PDF|important] [[Furcifer_PerCom24.pdf#page=3&selection=60,12,74,29&color=important|Furcifer_PerCom24, p.3]]
>  Furcifer aims to strike a balance between resource efficiency and predictive precision, spanning from the edge to the cloud, and catering to the comprehensive energy optimization needs of modern mobile computing. By minimizing the energy consumption of mobile devices based on the desired mean Average Precision (mAP ) score F P S rate, our solution represents a leap forward in realizing practical ubiquitous computer vision applications.


## Aspetti preliminari

> [!PDF|note] Edge Computing [[Furcifer_PerCom24.pdf#page=3&selection=101,0,111,27&color=note|Furcifer_PerCom24, p.3]]
> Edge Computing: The execution of the task on the edge server allows the use of high-performance models (e.g., large non-quantized models). However, the limited computing capabilities of ESs compared to cloud servers means that the server may struggle to serve a large number of task streams. Moreover, the need to transfer the input data to ES means that robust and high-capacity wireless channels are needed.

> [!PDF|note] Local Computing [[Furcifer_PerCom24.pdf#page=3&selection=123,0,162,42&color=note|Furcifer_PerCom24, p.3]]
> Local Computing: In settings where the task complexity is low enough to match the capabilities of the mobile device, then local execution of the algorithm is a viable option. A trade-off is struck between task performance, power consumption and frame rate. Importantly, LC performance is not dependent on the state of the wireless channel connecting the M D to the ES, or the network and server load. In this context, quantization assumes a pivotal role in reducing execution time and energy consumption and enabling the use of better performing models whose use would otherwise be impractical on resource-constrained devices with limited computational capabilities. However, LC implies high energy usage, which leads to a reduced battery lifespan.

> [!PDF|note] Split Computing [[Furcifer_PerCom24.pdf#page=3&selection=166,0,200,1&color=note|Furcifer_PerCom24, p.3]]
> Split (collaborative) Computing - SC: In SC, a subset of operations that would be executed on the ES is allocated to the mobile device. This subset often includes pre-processing operations, such as JPEG encoding and partial model inference altered to embed neural supervised encoding [33]. The objective is to decrease the amount of data to be transported over the wireless channel while minimizing the involvement of M D and possibly also decreasing the server load. This computing modality proves advantageous in settings where the communication channel’s reliability is compromised, bandwidth demands exceed channel capacity or computing demands exceed server capacity. SC is specifically designed to address this scenario by mitigating both channel usage and computation burden on the ES.


Nello specifico questo viene sottolineato perché la parte di compressione viene considerato uno degli aspetti principali in Furcifer
> [!PDF|note] [[Furcifer_PerCom24.pdf#page=3&selection=217,1,234,5&color=note|Furcifer_PerCom24, p.3]]
> However, when deploying an OD engine in a real-world setting, various factors such as camera resolution or scaling factor alterations come into play to determine the performance perceived by the application. Additional factors such as model quantization and image compression also play a significant role.

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=3&selection=284,1,288,61&color=important|Furcifer_PerCom24, p.3]]
> In our exploration of SC, we develop a specialized encoderdecoder architecture trained using supervised compression and Faster R-CNN as a teacher model. Our design is based on the model proposed in [37]. However, we optimized the original model by quantizing the encoder to FP16 and running it with an optimized inference engine.


> [!PDF|yellow] [[Furcifer_PerCom24.pdf#page=4&selection=148,14,163,1&color=yellow|Furcifer_PerCom24, p.4]]
> The results show that the best mAP performance is obtained using EC without JPEG compression - that is, the largest model running without image compression. Conversely, the maximum frame rate is achieved by a quantized version of the original model deployed on the M D.

> [!PDF|yellow] [[Furcifer_PerCom24.pdf#page=4&selection=166,37,173,6&color=yellow|Furcifer_PerCom24, p.4]]
> It should be pointed that this refers to the power consumed by M D and it does not take into account the overall total energy consumed by the whole system


## Design and Implementation
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=4&selection=208,49,215,43&color=important|Furcifer_PerCom24, p.4]]
>  the – time varying – state of the system, which is influenced by mobility and load dynamics, determines the best computing configuration. However, changing the computing modality in real-world deployments is technically non-trivial. Furcifer realizes an adaptation engine composed of highly effective containerized models whose activation is determined by a control module informed by comprehensive system monitoring

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=4&selection=217,25,219,29&color=important|Furcifer_PerCom24, p.4]]
> e container-based Service-Oriented Architecture (SOA) nature of Furcifer enables the independent deployment of each component.

### Monitoring
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=4&selection=247,1,281,45&color=important|Furcifer_PerCom24, p.4]]
> Energon focuses primarily on energy consumption and resource utilization in M Ds, while also providing insights into additional metrics, including network quality, packet transmission and drop rates, CPU usage for individual cores, storage utilization, GPU usage percentage, and temperature measurements from various regions of the board. Scraped metrics are made available through an HTTP endpoint that can be queried on demand by the orchestrator.

### Image Pulling
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=5&selection=87,0,91,23&color=important|Furcifer_PerCom24, p.5]]
> This registry stores well-tailored images optimized for each compatible device, which are cached for future use based on the specific task the device is assigned. For each device type, a subset of images shares identical interfaces with the operating system hypervisor

### Interface and Protocol
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=5&selection=208,0,224,6&color=important|Furcifer_PerCom24, p.5]]
> set target frame rate: This command sets the desired F P S rate for camera sampling based on dynamic requirements defined at the application level. Recognizing the direct correlation between higher F P S rates and increased power consumption, Furcifer intelligently conserves energy and network resources when higher frequency camera sampling is unnecessary, e.g., Vehicleto-Vehicle (V2V) cameras in low-traffic environments [39]).

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=5&selection=228,0,249,6&color=important|Furcifer_PerCom24, p.5]]
> set compression rate: If an EC configuration is used, the M D can opt to compress captured images before transmitting them to the ES for final detection. This message specifies the desired compression rate, controlling the balance between reduced compression for improved mAP score.


### Orchestrator
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=6&selection=25,13,31,57&color=important|Furcifer_PerCom24, p.6]]
> the orchestrator’s scaling and management protocols are controlled by the pareidolia policy. This policy evaluates the energy consumption metrics of the MD, as well as an array of context-sensitive metrics. This evaluation enables the orchestrator to dynamically adjust its operational strategy, seamlessly transitioning between two or more containers to optimize performance and resource utilization.


### SC Engine

In sostanza l'approccio qui è il seguente (da quello che sto capendo):
- Caso LC: computazione fatta solo su dispositivo locale e ok
- Caso EC: vengono usati dei meccanismi di compressione (eg JPEG) per ridurre la quantità di dati da trasferire da una parte all'altra
- Caso SC: vengono usati dei modelli di encoding nel device e di decoding nel edge node per poi eseguire il modello sull'edge node

Non abbiamo quindi un effettivo split del modello, ma un qualche tipo di codifica per ridurre i dati da trasferire. [[Furcifer_PerCom24.pdf#page=2&selection=262,0,276,11&color=note|Furcifer_PerCom24, p.2]].

In questo caso infatti stiamo considerando questi aspetti:
- Tempo di inferenza (e fin qui ok)
- Accuratezza
	- L'accuratezza dipende dal metodo di compressione. Come viene indicato anche nell'articolo, i formati tipo JPEG anche se forniscono un buon mezzo di compressione non sono ottimizzati per l'inferenza: un approccio basato su knowledge distillation quindi può risultare migliore in questo contesto

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=6&selection=56,14,61,26&color=important|Furcifer_PerCom24, p.6]]
>  we use a modified version of the knowledge distillation process adopted in SC2 Benchmark [37] to design a compact encoder optimized for constrained devices. This encoder serves a dual purpose: minimizing channel occupancy and effectively distributing computation load between mobile devices and the edge serve

> [!PDF|important] [[Furcifer_PerCom24.pdf#page=6&selection=90,1,124,0&color=important|Furcifer_PerCom24, p.6]]
> he optimized encoder heavily relies on quantization and channel compression to reduce execution time as much as possible. To enhance data compression, we strategically place a one-channel bottleneck in the initial layers of the feature extraction segment of the network. This choice leads to further data reduction, increasing the efficiency of the whole process. Additionally, we incorporate INT8 quantization at the end of the encoder. This quantization approach optimizes the representation of the data, contributing to both improved data compression and streamlined computation. The dynamic nature of the system is upheld by calculating the scaling factor and zero point on a per-image basis as they are processed. These values are then communicated to the decoder located at the ES, along with the resulting INT8 tensor from the encoder inference process.


### Pareidolia
> [!PDF|important] [[Furcifer_PerCom24.pdf#page=6&selection=236,26,266,36&color=important|Furcifer_PerCom24, p.6]]
> Each participating M D maintains a record of previously completed tasks. This historical context empowers the node to discern which computing strategy aligns best with the current system state by identifying analogous past scenarios. When a sufficiently similar context is detected, the ES intervention may not be required. Conversely, if an analogous context is not found, pertinent task details are shared with the ES to collaboratively determine the optimal model and computing configuration (EC, LC or SC) that best matches the current system stat


## Esperimenti




# [[gillis-icdcs21.pdf#page=1&selection=0,0,0,6&color=yellow|Gillis]]

> [!Quote] Abstract [[gillis-icdcs21.pdf#page=1&selection=28,1,56,22&color=red|gillis-icdcs21, p.1]]
> The increased use of deep neural networks has stimulated the growing demand for cloud-based model serving platforms. Serverless computing offers a simplified solution: users deploy models as serverless functions and let the platform handle provisioning and scaling. However, serverless functions have constrained resources in CPU and memory, making them inefficient or infeasible to serve large neural networks—which have become increasingly popular. 

> [!PDF|note] [[gillis-icdcs21.pdf#page=1&selection=94,56,101,1&color=note|gillis-icdcs21, p.1]]
> As inference is performed in real time with stringent SLOs (ServiceLevel Objectives), the model serving platforms must be made scalable to the changing workloads. This can be achieved by augmenting the traditional VM-based model serving with serverless functions:

> [!PDF|note] [[gillis-icdcs21.pdf#page=1&selection=108,0,134,31&color=note|gillis-icdcs21, p.1]]
> Unlike VMs, serverless functions have constrained resources in CPU and memory [10]–[12]. As very large DNNs are increasingly used for improved accuracy [13]–[15], using serverless functions to serve those models is inefficient or simply becomes infeasible with out-of-memory (OOM) errors

> [!PDF|important] [[gillis-icdcs21.pdf#page=1&selection=182,1,224,60&color=important|gillis-icdcs21, p.1]]
> n this paper, we propose Gillis1, a serverless-based model serving system that automatically explores parallel executions of DNN inference across multiple functions for faster inference and reduced per-function memory footprint. Our design follows the fork-join computing model: upon receiving an inference request, a master function is invoked to initiate multiple worker functions, each hosting a partition of the model. The master interacts with workers through stateless connections (function invocations). To reduce communications between master and workers, Gillis performs coarse-grained model partitioning. It fuses multiple consecutive layers of a DNN model into a single layer group. The model can have multiple layer groups. Gillis partitions each layer group for parallel execution. Such coarse-grained partitioning allows a function to compute all layers in a group locally, hence avoiding frequent synchronizations with the other functions.

## Background and Motivation

### Serverless Based
> [!PDF|note] [[gillis-icdcs21.pdf#page=2&selection=87,0,93,19&color=note|gillis-icdcs21, p.2]]
> Compared with VMs, serverless functions have a far shorter startup latency (e.g., 10s ms if warm-started) and can quickly scale out to a large number of instances to accommodate the surging inference requests in a short period of time. On the other hand, serverless functions are not well suited to serve stable workloads, as they have a high price per request [3], [23]. 

### Large Models
> [!PDF|note] [[gillis-icdcs21.pdf#page=3&selection=35,0,70,26&color=note|gillis-icdcs21, p.3]]
> Model Compression reduces the network parameters of a large DNN and creates a significantly smaller network that can run on a resource-constrained device (e.g., mobile and edge devices). Popular model compression techniques include network pruning [16] and weight quantization [17]. However, this approach usually results in reduced accuracy and requires careful model tuning or even retraining to minimize the accuracy loss, which is too laboring for developers [13]. 
> 
> 
> Model Partitioning Another promising approach is to divide a large neural network into multiple small partitions and run them in parallel [13], [18]–[21], [34], [35]. Compared with compression, model partitioning sacrifices no accuracy while accelerating the computation and reducing the perpartition resource footprint. We therefore choose it over model compression in our design.

> [!PDF|note] [[gillis-icdcs21.pdf#page=3&selection=71,0,82,22&color=note|gillis-icdcs21, p.3]]
> The key technique of this approach is tensor partitioning. In a DNN model, each layer takes an input tensor and computes an output tensor. An output tensor typically has multiple dimensions and can be parallelized in many different ways along those dimensions

![[gillis-icdcs21.pdf#page=3&rect=301,611,577,751&color=note|gillis-icdcs21, p.3]]

## Overview

### Workflow
> [!PDF|important] [[gillis-icdcs21.pdf#page=3&selection=274,2,282,57&color=important|gillis-icdcs21, p.3]]
> For each type of DNN layer, Gillis profiles its execution time in a single function. Gillis also profiles the function communication latency. Based on the profiling results, Gillis builds a performance model and uses it to predict the model execution time under various parallelization schemas.

> [!PDF|important] [[gillis-icdcs21.pdf#page=4&selection=74,32,86,20&color=important|gillis-icdcs21, p.4]]
>  In the model partitioning phase, Gillis accepts a serving model and generates a parallelization scheme to achieve the optimal inference latency (latency-optimal, §IV-B) or the minimum serving cost with SLO compliance (SLO-aware, §IV-C), using the performance model as a guideline

> [!PDF|important] [[gillis-icdcs21.pdf#page=4&selection=94,38,98,54&color=important|gillis-icdcs21, p.4]]
> Gillis supports periodically warming up functions by sending concurrent pings to the serverless platform [37]. As function instances stay active for a long time [38], the warm-up cost can be amortized by serving numerous inference queries and is hence negligible [3]

### Fork-Join
> [!PDF|important] [[gillis-icdcs21.pdf#page=4&selection=134,42,151,55&color=important|gillis-icdcs21, p.4]]
> A master function is triggered to run upon receiving an inference query. Following the computed partitioning scheme, the master asynchronously invokes multiple worker functions. Each worker computes a partition of the served model, returns the result to the master, and ends its execution. The master can also help to compute a partition if having sufficient memory, which can result in fewer workers and less cost. The master assembles the returned results from all workers into a full tensor, and may initiate more workers to continue parallelizing model execution.

### Coarse Grained Parallilization
> [!PDF|important] [[gillis-icdcs21.pdf#page=4&selection=184,11,196,44&color=important|gillis-icdcs21, p.4]]
> To reduce the communication overhead, Gillis instead performs coarse-grained parallelization: it combines multiple consecutive layers into a single group and parallelizes each group across serverless functions. All layers in a group are hence computed locally within a function

> [!PDF|important] [[gillis-icdcs21.pdf#page=4&selection=205,0,225,10&color=important|gillis-icdcs21, p.4]]
> First, our layer grouping is not limited to convolution layers, but applies to all. This enables more parallelization opportunities, yet significantly increases the search space for optimal grouping as a DNN model can have a large number of layers. To address this problem, we propose to merge consecutive element-wise layers (e.g., ReLU) and branch modules, if any. 

Anche qui quindi i branch vengono gestiti raggruppandoli in un maxi nodo.
> [!PDF|important] [[gillis-icdcs21.pdf#page=5&selection=41,54,45,25&color=important|gillis-icdcs21, p.5]]
>  we merge parallel branches into a single layer as shown in Fig. 5. Branch merging transforms a complex computation graph of a DNN model into a linear graph, substantially simplifying the partitioning strategy

> [!PDF|important] [[gillis-icdcs21.pdf#page=5&selection=55,3,70,11&color=important|gillis-icdcs21, p.5]]
> o meet this requirement, we determine if two consecutive layers can be grouped based on the dependency of their input and output tensors. Specifically, given two layers, if their output tensors have a local response to the input along the same dimensions, they can be group-parallelized along those dimensions.

> [!PDF|important] [[gillis-icdcs21.pdf#page=5&selection=84,1,100,24&color=important|gillis-icdcs21, p.5]]
> While layer grouping reduces the communication overhead, grouping too many layers can be inefficient, especially for those with convolution-like operators. As these operators (e.g., convolution and pooling) map multiple input elements to a single output, parallelizing the output tensor results in an overlap in the input partitions (Fig. 2a). As more layers are grouped, more overlaps are added, causing more redundant computations in the intermediate layers. Also, as the layer group grows larger, its partition may not fit into the memory of a single function. Parallelizing a layer group across too many functions can also be inefficient, as it may incur significant synchronization overhead in function communications, undermining the benefits of parallelization.

## RL Solution
> [!PDF|important] [[gillis-icdcs21.pdf#page=7&selection=89,10,102,33&color=important|gillis-icdcs21, p.7]]
> Our RL model has two agents, partitioner and placer, each of which is a two-layer neural network. The partitioner takes as input the DNN layers and determines how these layers are fused into groups and how each group is parallelized. Given the layer groups, the placer determines how partitions are placed on the master and workers, which works out a detailed function execution plan.



# Jellifish


# [[Splitting_2.pdf#page=1&selection=2,0,3,28&color=yellow|Learning the Optimal Path and DNN Partition for Collaborative Edge Inference]]

> [!Quote|] Abstract [[Splitting_2.pdf#page=1&selection=18,1,24,61|Splitting_2, p.1]]
> Recent advancements in Deep Neural Networks (DNNs) have catalyzed the development of numerous intelligent mobile applications and services. However, they also introduce significant computational challenges for resource-constrained mobile devices. To address this, collaborative edge inference has been proposed. This method involves partitioning a DNN inference task into several subtasks and distributing these across multiple network nodes. Despite its potential, most current approaches presume known network parameters—like node processing speeds and link transmission rates—or rely on a fixed sequence of nodes for processing the DNN subtasks. In this paper, we tackle a more complex scenario where network parameters are unknown and must be learned, and multiple network paths are available for distributing inference tasks.

> [!PDF|note] [[Splitting_2.pdf#page=1&selection=44,0,69,47&color=note|Splitting_2, p.1]]
> Growing number of contemporary applications and services, such as augmented reality/virtual reality, face recognition, and speech assistant, demand real-time inferencing of deep neural networks (DNNs), which can effectively extract high-level features from the raw data at high computational complexity cost [1], [2]. In many cases of practical importance, these applications often are run at resource-constrained embedded devices, such as mobile phones, wearables and IoT devices in next-generation networks [3], [4]. However, it is very challenging to compute the DNN inference tasks at resource-constrained devices due to the limited computation and battery capacity. Therefore, it is crucial to leverage external computation resources to realize the full potential and benefits of future devicebased artificial intelligence (AI) applications

> [!PDF|note] [[Splitting_2.pdf#page=1&selection=88,18,92,61&color=note|Splitting_2, p.1]]
> By combining the capabilities of both ondevice processing and computation offloading, resourceconstrained devices can delegate all or part of their inference workload to network edge devices, such as cellular base stations, access points, or peer devices within the same net-

> [!PDF|note] [[Splitting_2.pdf#page=1&selection=163,7,165,57&color=note|Splitting_2, p.1]]
> On the other hand, these devices can also provide alternative paths between the mobile device and the edge server, particularly useful when the direct communication link experiences poor conditions.

Key challenges:
1. compared to partitioning the DNN between just two devices, the decision space expands significantly. This expansion involves not only selecting an appropriate communication path between the mobile device and the edge server but also assigning DNN layers to the devices along the path to distribute the DNN inference workload effectively
2. he end-to-end inference latency is contingent on the computing speed of the devices and the transmission speed of the links along the selected path. These speeds fluctuate over time and may not be known beforehand when making collaborative inference decisions
3. switching devices for collaborative edge inference necessitates reloading and setting up the DNN models in the device memory, which introduces overhead and can impact the overall inference latency. Thus, frequent device and path switching is undesirable.

> [!PDF|important] [[Splitting_2.pdf#page=2&selection=33,0,40,6&color=important|Splitting_2, p.2]]
> In this paper, we study the joint optimization of path selection and DNN layer assignment in collaborative edge inference across multiple devices. Specifically, we explore scenarios where multiple paths exist between a source node (i.e., the mobile device) and a destination node (i.e., the edge server), with a focus on optimizing path selection and DNN layer assignments to minimize end-to-end inference delay.


## System Model

Questo aspetto è abbastanza simile alla nostra formulazione: ho un tot di richieste che vengono formulate e ognuna ha la sua DNN associata.
> [!PDF|important] [[Splitting_2.pdf#page=3&selection=164,35,186,53&color=important|Splitting_2, p.3]]
>  In addition to the edge server, there are other available devices in the network, functioning as relays and creating multiple possible communication paths between the mobile device and the edge server. The mobile device needs to sequentially process a series of DNN inference tasks, denoted by the task index t = 1, . . . , T . Each task t is a DNN inference task using DNNt, which might change over time or remain constant. Each inference task utilizes input data collected by the mobile device and aims to deliver the inference results to the edge server for downstream applications/services.

La rete di nodi sottostanti è considerata un DAG, quindi non è considerata la possibilità di tornare indietro: l'output evidentemente viene mantenuto a livello dell'edge server finale. Nel nostro caso invece considerare un grafo generico permette di considerare anche percorsi alternativi: ad esempio una situazione in cui in fase di elaborazione ci si allontana via via dal device e mano mano che ci si avvicina all'output ci si riavvicina al device.
> [!PDF|important] [[Splitting_2.pdf#page=3&selection=254,0,261,2&color=important|Splitting_2, p.3]]
> The communication network linking a mobile device with an edge server can be represented by an acyclic directed graph, denoted as ⟨V, E⟩

> [!PDF|important] [[Splitting_2.pdf#page=3&selection=402,2,440,12&color=important|Splitting_2, p.3]]
> The layer assignment function fg adheres to two key properties for collaborative inference: • Many-to-One: Multiple consecutive DNN layers may be processed by a single node. • Non-decreasing: If a layer l is assigned to a node v, subsequent layers cannot be assigned to any nodes preceding v in the path.

![[Splitting_2.pdf#page=3&rect=304,268,571,375&color=important|Splitting_2, p.3]]

Viene considerata la possibilità che vi sia un "attacco" in senso lato su un link: questo rappresenta in generale l'incertezza del canale di comunicazione.
> [!PDF|important] [[Splitting_2.pdf#page=3&selection=619,0,624,39&color=important|Splitting_2, p.3]]
> Besides the inherent uncertainties stemming from variations in computing speed and transmission rates, the collaborative edge inference system may encounter additional non-stochastic uncertainties. We consider the potential presence of an adversary in the system, which aims to select paths to attack during each time slot. 

> [!PDF|important] [[Splitting_2.pdf#page=4&selection=152,0,168,39&color=important|Splitting_2, p.4]]
> Objective: The goal for the learner is to minimize the total inference delay while concurrently reducing the cumulative switching costs across T inference tasks (t = 1, ..., T ). This involves strategically choosing the optimal path and corresponding DNN layer assignment for each task to maximize the following objective function:

> [!PDF|important] [[Splitting_2.pdf#page=4&selection=209,30,214,39&color=important|Splitting_2, p.4]]
> On one hand, the learner must choose the optimal path considering the state of the communication network and striving to avoid attackers as effectively as possible. On the other hand, the learner must determine the best DNN layer assignment for the chosen path to minimize total inference delay.

> [!PDF|note] [[Splitting_2.pdf#page=4&selection=284,0,292,57&color=note|Splitting_2, p.4]]
> Our first result highlights the splitting positions within a DNN that would never emerge in the optimal layer assignment based solely on information about the DNN architecture. This means that even without knowledge of node processing speeds and link transmission speeds, and without solving the optimal layer assignment problem, it is possible to pre-group DNN layers into blocks. Consequently, rather than conducting DNN layer assignment directly, we can alternatively pursue DNN block assignment.

Questa assunzione di base è un'assunzione forte: non c'è nessuna garanzia che questo avvenga in un modello generico. Si tratta comunque di un teorema che potrebbe essere usato per descrivere il fatto che lo spazio di ricerca seppur grande presenta un numero alto di combinazioni che non portano a niente. Una possibile alternativa potrebbe essere usare un approccio simile a questo e far eseguire l'ottimizzazione sui blocchi, prendendo come flops la somma dei layer nel blocco.
> [!PDF|note] [[Splitting_2.pdf#page=4&selection=294,1,319,31&color=note|Splitting_2, p.4]]
> Proposition 1. Assume that among all layer outputs and the input, the data size of the output layer is the smallest, i.e., sL < sl, ∀l ∈ {0, ..., L − 1}. In the optimal layer assignment, the data sizes at the intermediate output splitting points form a non-increasing sequence.

In generale comunque anche qui non viene fatta una gestione effettiva dei branch e dei rami paralleli del modello: ci si limita a considerare il tutto come un blocco e dividere i blocchi. C'è da dire che nella parte sperimentale è sottolineato come sono considerati ResNet50 e YoLo (credo versione 1 - confermato dal paper a cui si rimanda il link) che sono dei modelli tendenzialmente lineari: non è quindi necessario gestire la complessità di rami paralleli.

![[Splitting_2.pdf#page=5&rect=52,480,290,618&color=note|Splitting_2, p.5]]

Anche questo dipende dalla situazione. Nel contesto del paper può avere senso: si sta assumendo un DAG, quindi l'output non può mai tornare al generatore, ma nel nostro caso è esattamente il contrario. 
> [!PDF|note] [[Splitting_2.pdf#page=5&selection=193,0,197,31&color=note|Splitting_2, p.5]]
> Proposition 2. In an optimal layer assignment, the processing speeds of nodes assigned with layer processing tasks form a non-decreasing sequence.


# [[Tuli_Splitting.pdf#page=1&selection=0,0,0,10|SplitPlace]]

> [!Quote|] Abstract [[Tuli_Splitting.pdf#page=1&selection=10,1,17,136|Tuli_Splitting, p.1]]
> n recent years, deep learning models have become ubiquitous in industry and academia alike. Deep neural networks can solve some of the most complex pattern-recognition problems today, but come with the price of massive compute and memory requirements. This makes the problem of deploying such large-scale neural networks challenging in resource-constrained mobile edge computing platforms, specifically in mission-critical domains like surveillance and healthcare. To solve this, a promising solution is to split resource-hungry neural networks into lightweight disjoint smaller components for pipelined distributed processing. At present, there are two main approaches to do this: semantic and layer-wise splitting. The former partitions a neural network into parallel disjoint models that produce a part of the result, whereas the latter partitions into sequential models that produce intermediate results. However, there is no intelligent algorithm that decides which splitting strategy to use and places such modular splits to edge nodes for optimal performance.

> [!PDF|note] [[Tuli_Splitting.pdf#page=1&selection=64,9,66,51&color=note|Tuli_Splitting, p.1]]
> to provide high accuracy, such neural models are becoming increasingly demanding in terms of data and compute power, resulting in many challenging problems.

> [!PDF|note] [[Tuli_Splitting.pdf#page=1&selection=73,0,86,19&color=note|Tuli_Splitting, p.1]]
> Recently, application demands have shifted from either high-accuracy or low-latency to both of these together, termed as HALL (high-accuracy and low-latency) service delivery [2]. Given the prevalence and demand of DNN inference, serving them on a public cloud with tight bounds of latency, throughput and cost is becoming increasingly challenging [9]. In this regard, recent paradigms like mobile edge computing seem promising. Such approaches allow a robust and low-latency deployment of Internet of Things (IoT) applications close to the edge of the network. Specifically, to solve the problem of providing HALL services, recent work proposes to integrate large-scale deep learning models with modern frameworks like edge computing [9], [10], [11]

> [!PDF|note] [[Tuli_Splitting.pdf#page=1&selection=90,0,96,9&color=note|Tuli_Splitting, p.1]]
> Another challenge of using edge computing is that mobile edge devices face severe limitations in terms of computational and memory resources as they rely on low power energy sources like batteries, solar or other energy scavenging methods [12], [13]. This is not only because of the requirement of low cost, but also the need for mobility in such nodes [5]

> [!PDF|note] [[Tuli_Splitting.pdf#page=2&selection=23,49,27,47&color=note|Tuli_Splitting, p.2]]
> in CloudAI where AI systems are deployed on cloud machines, the high communication latency leads to high average response times, making it unsuitable for latency-critical applications like healthcare, gaming and augmented reality [

> [!PDF|important] [[Tuli_Splitting.pdf#page=2&selection=121,22,133,25&color=important|Tuli_Splitting, p.2]]
> SplitPlace is the first splitting policy that dynamically decides between semantic and layer-wise splits to optimize both inference accuracy and the SLA violation rate. This decision is taken for each incoming task and remains unmodified until the execution of all split fragments of that task are complete. The idea behind the proposed splitting policy is to decide for each incoming task whether to use the semantic or layer-wise splitting strategy based on its SLA demands.

## Background

> [!PDF|note] [[Tuli_Splitting.pdf#page=3&selection=12,35,29,43&color=note|Tuli_Splitting, p.3]]
> Semantic splitting divides the network weights into a hierarchy of multiple groups that use a different set of features (different colored models in Fig. 1). Here, the neural network is split based on the data semantics, producing a tree structured model that has no connection among branches of the tree, allowing parallelization of input analysis [16]. Due to limited information sharing among the neural network fragments, the semantic splitting scheme gives lower accuracy in general. Semantic splitting requires a separate training procedure where publicly available pre-trained models cannot be used. This is because a pre-trained standard neural network can be split layer wise without affecting output semantics. For semantic splitting we would need to first split the neural network based on data semantics and re-train the model. However, semantic splitting provides parallel task processing and hence lower inference times, more suitable for mission-critical tasks like healthcare and surveillance.

> [!PDF|note] [[Tuli_Splitting.pdf#page=3&selection=29,44,40,28&color=note|Tuli_Splitting, p.3]]
> Layer-wise splitting divides the network into groups of layers for sequential processing of the task input, shown as different colored models in Fig. 1. Layer splitting is easier to deploy as pre-trained models can be just divided into multiple layer groups and distributed to different mobile edge nodes. However, layer splits require a semi-processed input to be forwarded to the subsequent edge node with the final processed output to be sent to the user, thus increasing the overall execution time. Moreover, layer-wise splitting gives higher accuracy compared to semantic splitting.

## Related Works
> [!PDF|note] [[Tuli_Splitting.pdf#page=3&selection=98,37,105,14&color=note|Tuli_Splitting, p.3]]
> Recently, architectures like BottleNet and Bottlenet++ have been proposed [37], [38] to enable DNN inference on mobile cloud environments and reduce data transmission times. BottleNet++ compresses the intermediate layer outputs before sending them to the cloud layer. It uses a model re-training approach to prevent the inference being adversely impacted by the lossy compression of data. 

Questo è il nostro caso!
> [!PDF|note] [[Tuli_Splitting.pdf#page=4&selection=32,6,42,51&color=note|Tuli_Splitting, p.4]]
> In heterogeneous edge-cloud environments, it is fairly straightforward to split the network into two or three fragments each being deployed in a mobile device, edge node or a cloud server. Based on the SLA, such methods provide early-exits if the turnaround time is expected to be more than the SLA deadline. This requires a part of the inference being run at each layer of the network architecture instead of traditionally executing it on the cloud server. Other recent methods aim at exploiting the resource heterogeneity in the same network layer by splitting and placing DNNs based on user demands and edge worker capabilities 


## System Model and Problem Formulation
> [!PDF|important] [[Tuli_Splitting.pdf#page=5&selection=5,0,36,33&color=important|Tuli_Splitting, p.5]]
> Some worker nodes are assumed to be mobile, whereas others are considered to be fixed in terms of their geographical location. In our formulation, we consider mobility only in terms of the variations in terms of the network channels and do not consider the worker nodes or users crossing different networks

> [!PDF|important] [[Tuli_Splitting.pdf#page=5&selection=44,35,48,45&color=important|Tuli_Splitting, p.5]]
> The broker periodically measure utilizations of CPU, RAM, Bandwidth and Disk for each task in the system. The broker is trusted with this information such that it can make informed resource management decisions to optimize QoS

## Splitting
> [!PDF|note] [[Tuli_Splitting.pdf#page=5&selection=264,0,266,47&color=note|Tuli_Splitting, p.5]]
> We partition the problem into two sub-problems of deciding the optimal splitting strategy for input tasks and that of placement of active containers in edge workers 

> [!PDF|note] [[Tuli_Splitting.pdf#page=6&selection=13,0,29,7&color=note|Tuli_Splitting, p.6]]
> The main idea behind the layer-wise split design is first to divide neural networks into multiple independent splits, classify these splits in preliminary, intermediate and final neural network layers and distribute them across different nodes based on the node capabilities and network hierarchy. This exploits the fact that communication across edge nodes in the LAN with few hop distances is very fast and has low latency and jitter [51]. Moreover, techniques like knowledge distillation can be further utilized to enhance the accuracy of the results obtained by passing the input through these different classifiers. However, knowledge distillation needs to be applied at the training stage, before generating the neural network splits. As there are many inputs in our assumed large-scale deployment, the execution can be performed in a pipelined fashion to further improve throughput over and above the low response time of the nodes at the edge of the network

> [!PDF|note] [[Tuli_Splitting.pdf#page=6&selection=29,9,52,26&color=note|Tuli_Splitting, p.6]]
> For the semantic split, we divide the network weights into a set or a hierarchy of multiple groups that use disjoint sets of features. This is done by making assignment decisions of network parameters to edge devices at deployment time. This produces a tree-structured network that involves no connection between branched sub-trees of semantically disparate class groups. Each sub-group is then allocated to an edge node. The input is either broadcasted from the broker or forwarded in a ring-topology to all nodes with the network split corresponding to the input task. We use standard layer [32] and semantic splitting [16] methods as discussed in Section 2.

> [!PDF|note] [[Tuli_Splitting.pdf#page=6&selection=82,0,93,0&color=note|Tuli_Splitting, p.6]]
> This sharing of containers for each splitting strategy and dataset type are transferred to the worker nodes is performed at the start of the run. At run-time, only the decision of which split fragment to be used is communicated to the worker nodes, which executes a container from the corresponding image. The placement of task on each worker is based on the resource availability, computation required to be performed in each section and the capabilities of the nodes (obtained by the Resource Monitor


## Policy
> [!PDF|important] [[Tuli_Splitting.pdf#page=7&selection=2,0,26,1&color=important|Tuli_Splitting, p.7]]
> > we employ a Multi-Armed Bandit model to dynamically enforce the decision using external reward signals. 1 Our solution for the second sub-problem of split placement uses a reinforcement-learning based approach that specifically utilizes a surrogate model to optimize the placement decision (agnostic to the specific implementation). 2

> [!PDF|important] [[Tuli_Splitting.pdf#page=7&selection=28,0,32,25&color=important|Tuli_Splitting, p.7]]
> This two-stage approach is suboptimal since the response time of the splitting decision depends on the placement decision. In case of large variation in terms of the computational resources, it is worth exploring joint optimization of both decisions.

> [!PDF|important] [[Tuli_Splitting.pdf#page=7&selection=57,18,60,7&color=important|Tuli_Splitting, p.7]]
>  the response time of an application depends primarily on the splitting choice, layer or semantic, making it a crucial factor for SLA deadline based decision making.

## Esperimenti
Potrebbe essere interessante per simulare scenari di movimento e/o scenari variabili.
> [!PDF|yellow] [[Tuli_Splitting.pdf#page=10&selection=158,0,161,27&color=yellow|Tuli_Splitting, p.10]]
> we use the latency and bandwidth parameters of workers from the traces generated using the Simulation of Urban Mobility (SUMO) tool [67] that emulates mobile vehicles in a city like environment.

Da qui si potrebbero prendere i consumi energetici
> [!PDF|yellow] [[Tuli_Splitting.pdf#page=10&selection=240,0,243,4&color=yellow|Tuli_Splitting, p.10]]
> The power consumption models are taken from the Standard Performance Evaluation Corporation (SPEC) benchmarks repository

Da qui si potrebbero ricavare le MIPS / FLOPS della macchina su cui ci si trova
> [!PDF|yellow] [[Tuli_Splitting.pdf#page=10&selection=247,4,253,16&color=yellow|Tuli_Splitting, p.10]]
> Million-Instruction-per-Second (MIPS) of all VMs are computed using the perf-stat8 tool on the SPEC

Anche qui vengono usati reti abbastanza semplici
> [!PDF|yellow] [[Tuli_Splitting.pdf#page=11&selection=19,0,21,37&color=yellow|Tuli_Splitting, p.11]]
> Motivated from prior work [32], we use three families of popular DNNs as the benchmarking models: ResNet50-V2 [69], MobileNetV2 [70] and InceptionV3 [71]

Controllare questi altri lavori: si tratta comunque di splitting Semantico della rete, non per livelli.
> [!PDF|yellow] [[Tuli_Splitting.pdf#page=11&selection=93,10,94,37&color=yellow|Tuli_Splitting, p.11]]
>  We use the implementation of neural network splitting from prior work [16], [32].


# [[Survey_Splitting.pdf#page=1&selection=2,14,2,44&color=red|Survey and Research Challenges]]

> [!PDF|note] [[Survey_Splitting.pdf#page=1&selection=11,0,13,104&color=note|Survey_Splitting, p.1]]
> Mobile devices such as smartphones and autonomous vehicles increasingly rely on deep neural networks (DNNs) to execute complex inference tasks such as image classification and speech recognition, among others. However, continuously executing the entire DNN on mobile devices can quickly deplete their battery.

> [!PDF|note] [[Survey_Splitting.pdf#page=1&selection=16,8,24,62&color=note|Survey_Splitting, p.1]]
> Recently, approaches based on split computing (SC) have been proposed, where the DNN is split into a head and a tail model, executed respectively on the mobile device and on the edge server. Ultimately, this may reduce bandwidth usage as well as energy consumption.

> [!PDF|note] [[Survey_Splitting.pdf#page=1&selection=24,63,31,16&color=note|Survey_Splitting, p.1]]
> Another approach, called early exiting (EE), trains models to embed multiple “exits” earlier in the architecture, each providing increasingly higher target accuracy.

## Introduzione
> [!PDF|note] [[Survey_Splitting.pdf#page=2&selection=12,1,22,1&color=note|Survey_Splitting, p.2]]
> As DL-based classifiers improve their predictive accuracy, mobile applications such as speech recognition in smartphones [20, 45], real-time unmanned navigation [105], and drone-based surveillance [129, 170] are increasingly using DNNs to perform complex inference tasks. However, state-of-the-art DNN models present computational requirements that cannot be satisfied by the majority of the mobile devices available today.

> [!PDF|note] [[Survey_Splitting.pdf#page=2&selection=31,49,34,22&color=note|Survey_Splitting, p.2]]
> Notably, the execution of such complex models significantly increases energy consumption. While lightweight models specifically designed for mobile devices exist [122, 138], the reduced computational burden usually comes to the detriment of the model accuracy.

> [!PDF|note] [[Survey_Splitting.pdf#page=2&selection=45,6,46,96&color=note|Survey_Splitting, p.2]]
> On the other hand, due to excessive end-to-end latency, cloud-based approaches are hardly applicable in most of the latency-constrained applications where mobile devices usually operate.

> [!PDF|note] [[Survey_Splitting.pdf#page=2&selection=52,0,65,41&color=note|Survey_Splitting, p.2]]
> edge computing (EC) approaches [10, 88] have attempted to address the “latency vs. computation” conundrum by completely offloading the DNN execution to servers located very close to the mobile device, i.e., at the “edge” of the network. However, canonical EC does not consider that the quality of wireless links—although providing high throughput on average—can suddenly fluctuate due to the presence of erratic noise and interference patterns, which may impair performance in latency-bound applications

> [!PDF|note] [[Survey_Splitting.pdf#page=3&selection=20,1,37,84&color=note|Survey_Splitting, p.3]]
> The proliferation of DL-based mobile applications in the IoT and 5G landscapes implies that techniques such as SC and EE are not simply “nice-to-have” features, but will become fundamental computational components in the years to come.

## Overview ([[Survey_Splitting.pdf#page=3&selection=105,0,105,63|p.3]])

> ([[Survey_Splitting.pdf#page=3&selection=135,36,137,22&color=note|p.3]])
> Split computing and early-exit approaches are contextualized in a setting where the system is composed of a mobile device and an edge server interconnected via a wireless channel

Limiti su :
> [!PDF|note] [[Survey_Splitting.pdf#page=3&selection=161,0,177,79&color=note|Survey_Splitting, p.3]]
> Resources: (1) the computational capacity (roughly expressed as number operations per second) Cmd and Ces of the mobile device and edge server, respectively, and (2) the capacity ϕ, in bits per second, of the wireless channel connecting the mobile device to the edge server

> [!PDF|note] [[Survey_Splitting.pdf#page=3&selection=179,0,192,1&color=note|Survey_Splitting, p.3]]
> Performance: (1) the absolute of average value of the time from the generation of x to the availability of y, and (2) the degradation of the “quality” of the output y.

![[Survey_Splitting.pdf#page=4&rect=50,394,440,646&color=note|Survey_Splitting, p.4]]

> [!PDF|note] [[Survey_Splitting.pdf#page=4&selection=8,79,12,84&color=note|Survey_Splitting, p.4]]
> As a consequence, if part of the workload is allocated to the mobile device, then the execution time increases, while the battery lifetime decreases. However, as explained later, the workload executed by the mobile device may result in a reduced amount of data to be transferred over the wireless channel, possibly compensating for the larger execution time and leading to smaller end-to-end delays.

### [[Survey_Splitting.pdf#page=4&selection=16,0,16,24&color=yellow|Local and Edge Computing]]

> [!PDF|note] [[Survey_Splitting.pdf#page=4&selection=18,55,34,84&color=note|Survey_Splitting, p.4]]
> In local computing (LC), the function M (x) is entirely executed by the mobile device. This approach eliminates the need to transfer data over the wireless channel. However, the complexity of the best-performing DNNs most likely exceeds the computing capacity and energy consumption available at the mobile device

> [!PDF|note] [[Survey_Splitting.pdf#page=4&selection=45,31,47,51&color=note|Survey_Splitting, p.4]]
> Besides designing lightweight neural models executable on mobile devices, the widely used techniques to reduce the complexity of models are knowledge distillation [46] and model pruning/quantization [55, 73], 

> [!PDF|note] [[Survey_Splitting.pdf#page=4&selection=54,0,75,11&color=note|Survey_Splitting, p.4]]
> In EC, the input x is transferred to the edge server, which then executes the original model M (x). In this approach, which preserves full accuracy, the mobile device is not allocated computing workload, but the full input x needs to be transferred to the edge server. This may lead to an excessive end-to-end delay in degraded channel conditions and erasure of the task in extreme conditions.

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=4,27,55,47&color=note|Survey_Splitting, p.5]]
> The distance d (x, ˆx) defines the performance of the encoding-decoding process ˆx = G (F (x)), a metric that is separate from, but may influence, the accuracy loss of M ( ˆx) with respect to M (x), that is, of the model executed with the reconstructed input with respect to the model executed with the original input. Clearly, the encoding/decoding functions increase the computing load at both the mobile device and edge server side. A broad range of different compression approaches exists ranging from low-complexity traditional compression (e.g., JPEG compression for images in EC [101]) to neural compression models [4, 5, 162]

### Split Computing and Early Exiting
> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=72,0,74,7&color=note|Survey_Splitting, p.5]]
> SC aims at achieving the following goals: (1) the computing load is distributed across the mobile device and edge server and (2) establishes a task-oriented compression to reduce data transfer delays.

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=95,7,139,12&color=note|Survey_Splitting, p.5]]
> Early implementations of SC select a layer  and divide the model M (·) to define the head and tail submodels z =MH (x) and ˆy=MT (z ), executed at the mobile device and edge server, respectively

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=173,0,186,0&color=note|Survey_Splitting, p.5]]
> The transmission time of z may be larger or smaller compared to that of transmitting the input x, depending on the size of the tensor z 

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=197,41,221,20&color=note|Survey_Splitting, p.5]]
> More recent SC frameworks introduce the notion of bottleneck to achieve in-model compression toward the global task [90]. As formally described in the next section, a bottleneck is a compression point at one layer in the model, which can be realized by reducing the number of nodes of the target layer and/or by quantizing its output. We note that as SC realizes a task-oriented compression, it guarantees a higher degree of privacy compared to EC. In fact, the representation may lack information needed to fully reconstruct the original input data.

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=222,0,223,92&color=note|Survey_Splitting, p.5]]
> Another approach to enable mobile computing is referred to as EE. The core idea is to create models with multiple “exits” across the model, where each exit can produce the model output.

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=226,0,309,26&color=note|Survey_Splitting, p.5]]
> Formally, we can define a sequence of models Mi and Bi , i = 1, . . . , N . Model Mi takes as input zi−1 (the output of model Mi−1 ) and outputs zi , where we set z0 = x. The branch models Bi take as input zi and produce the estimate of the desired output yi . Thus, the concatenation of M1, . . . , MN results in an output analogous to that of the original model. Intuitively, the larger the number of models used to produce the output yi , the better the accuracy.

> [!PDF|note] [[Survey_Splitting.pdf#page=5&selection=315,78,327,10&color=note|Survey_Splitting, p.5]]
> Each sample will be sequentially analyzed by concatenations of Mi and Bi sections until a predefined confidence level is reached


## [[Survey_Splitting.pdf#page=6&selection=6,0,6,10|BACKGROUND]]

### Lightweight models

> [!PDF|note] [[Survey_Splitting.pdf#page=6&selection=17,0,20,7&color=note|Survey_Splitting, p.6]]
> From a conceptual perspective, the design of small deep learning models is one of the simplest ways to reduce inference cost. However, there is a tradeoff between model complexity and model accuracy, which makes this approach practically challenging when aiming at high model performance. 

> [!PDF|note] [[Survey_Splitting.pdf#page=6&selection=20,7,21,22&color=note|Survey_Splitting, p.6]]
> The MobileNet series [47, 48, 122] is one among the most popular lightweight models for computer vision tasks,

> [!PDF|note] [[Survey_Splitting.pdf#page=6&selection=28,16,30,11&color=note|Survey_Splitting, p.6]]
> The largest variant of MobileNetV3, MobileNetV3-Large 1.0, achieves a comparable accuracy to ResNet-34 [43] for the ImageNet dataset, while reducing by about 75% the model parameters.

### Model Compression

> [!PDF|note] [[Survey_Splitting.pdf#page=6&selection=59,0,61,86&color=note|Survey_Splitting, p.6]]
> A different approach to produce small DNN models is to “compress” a large model. Model pruning and quantization [38, 39, 55, 79] are the dominant model compression approaches. The former removes parameters from the model, while the latter uses fewer bits to represent them.

> [!PDF|note] [[Survey_Splitting.pdf#page=6&selection=70,0,71,93&color=note|Survey_Splitting, p.6]]
> Knowledge distillation [8, 46] is another popular model compression method. While model pruning and quantization make trained models smaller, the concept of knowledge distillation is to provide outputs extracted from the trained model (called “teacher”) as informative signals to train smaller models (called “student”) in order to improve the accuracy of predesigned small models. Thus, the goal of the process is that of distilling knowledge of a trained teacher model into a smaller student model for boosting accuracy of the smaller model without increasing model complexity.

## Split Computing Survey
> [!PDF|note] [[Survey_Splitting.pdf#page=7&selection=35,11,43,1&color=note|Survey_Splitting, p.7]]
>  They can be categorized into either (1) without network modification or (2) with bottleneck injection.

### Without

Controllare lo studio che riferiscono (Neurosurgeon)
> [!PDF|note] [[Survey_Splitting.pdf#page=7&selection=98,3,105,12&color=note|Survey_Splitting, p.7]]
> To the best of our knowledge, Kang et al. [60] proposed the first SC approach (called “Neurosurgeon”), which searches for the best partitioning layer in a DNN model for minimizing total (end-to-end) latency or energy consumption.

> [!PDF|note] [[Survey_Splitting.pdf#page=7&selection=112,0,114,23&color=note|Survey_Splitting, p.7]]
> Interestingly, their experimental results show that the best partitioning (splitting) layers in terms of energy consumption and total latency for most of the considered models result in either their input or output layers.

Controllare questo studio e il modo in cui viene valutata la quantizzazione e il suo effetto (se viene effettivamente valutato)
> [!PDF|note] [[Survey_Splitting.pdf#page=8&selection=222,33,224,47&color=note|Survey_Splitting, p.8]]
> Li et al. [73] discussed best splitting point in DNN models to minimize inference latency and showed that quantized DNN models did not degrade accuracy compared to the (pre-quantized) original models

> [!PDF|note] [[Survey_Splitting.pdf#page=8&selection=227,0,231,91&color=note|Survey_Splitting, p.8]]
> Eshratifar et al. [25] propose JointDNN for collaborative computation between the mobile device and cloud and demonstrate that using either local computing only or cloud computing only is not an optimal solution in terms of inference time and energy consumption. Different from [60], they consider not only discriminative deep learning models (e.g., classifiers) but also generative deep learning models and autoencoders as benchmark models in their experimental evaluation.

> [!PDF|note] [[Survey_Splitting.pdf#page=9&selection=92,43,98,74&color=note|Survey_Splitting, p.9]]
>  Concerning metrics, many of the studies in Table 1 do not discuss task-specific performance metrics such as accuracy. This is in part because the proposed approaches do not modify the input or intermediate representations in the models (i.e., the final prediction will not change). On the other hand, Choi and Bajić [13], Cohen et al. [16], and Li et al. [73] introduce lossy compression techniques to intermediate stages in DNN models, which may affect the final prediction results. Thus, discussing the tradeoff between compression rate and task-specific performance metrics would be essential for such studies.

Questo aspetto è significativo: nel mio caso mi sono concentrato soprattutto su modello YOLO che sono particolarmente complessi,
> [!PDF|note] [[Survey_Splitting.pdf#page=9&selection=103,11,113,11&color=note|Survey_Splitting, p.9]]
> many of the models considered in such studies have weak performance compared with state-of-the-art models and complexity within reach of modern mobile devices. Specific to image classification tasks, most of the models considered in the studies listed in Table 1 are more complex and/or the accuracy is comparable to or lower than that of lightweight baseline models such as MobileNetV2 [122] and MnasNet [138]. Thus, in future work, more accurate models should be considered to discuss the performance tradeoff and further motivate SC approaches.


### Bottleneck Injection

> [!PDF|note] [[Survey_Splitting.pdf#page=9&selection=127,22,140,31&color=note|Survey_Splitting, p.9]]
> There are a few trends observed from their experimental results: (1) communication delay to transfer data from the mobile device to edge server is a key component in SC to reduce total inference time; (2) all the neural models they considered for NLP tasks are relatively small (consisting of only a few layers), which potentially resulted in finding that the output layer is the best partition point (i.e., local computing) according to their proposed approach; and (3) similarly, not only DNN models they considered (except VGG [128]) but also the size of the input data to the models (see Table 2) are relatively small, which gives more advantage to EC (fully offloading computation).

> [!PDF|note] [[Survey_Splitting.pdf#page=9&selection=140,48,151,8&color=note|Survey_Splitting, p.9]]
> it highlights that complex CV tasks requiring large (high-resolution) images for models to achieve high accuracy such as ImageNet and COCO datasets would be essential to discuss the tradeoff between accuracy and execution metrics to be minimized (e.g., total latency, energy consumption) for SC studies.

> [!PDF|note] [[Survey_Splitting.pdf#page=9&selection=151,30,165,54&color=note|Survey_Splitting, p.9]]
>  straightforward SC approaches like Kang et al. [60] rely on the existence of natural bottlenecks—that is, intermediate layers whose output z tensor size is smaller than the input—inside the model

> [!PDF|note] [[Survey_Splitting.pdf#page=10&selection=6,0,10,6&color=note|Survey_Splitting, p.10]]
> Some models, such as AlexNet [64], VGG [128], and DenseNet [51], possess such layers [90]. However, recent DNN models such as ResNet [43], Inception-v3 [136], Faster R-CNN [117], and Mask R-CNN [42] do not have natural bottlenecks in the early layers; that is, splitting the model would result in compression only when assigning a large portion of the workload to the mobile device

> [!PDF|note] [[Survey_Splitting.pdf#page=10&selection=14,2,25,62&color=note|Survey_Splitting, p.10]]
> For these reasons, introducing artificial bottlenecks to DNN models by modifying their architecture is a recent trend and has been attracting attention from the research community. Since the main role of such encoders in SC is to compress intermediate features rather than to complete inference, the encoders usually consist of only a few layers. 

> [!PDF|note] [[Survey_Splitting.pdf#page=10&selection=154,3,215,26&color=note|Survey_Splitting, p.10]]
> n worlds, the two first sections of the modified model transform the input x into a version of the output of the th layer via the intermediate representation z, thus functioning as encoder/decoder functions. The model is split after the first section; that is, ME is the head model, and the concatenation of MD and MT is the tail model. Then, the tensor z is transmitted over the channel. The objective of the architecture is to minimize the size of z to reduce the communication time while also minimizing the complexity of ME (that is, the part of the model executed at the—weaker—mobile device) and the discrepancy between y and ˆy. The layer between ME and MD is the injected bottleneck

Ricontrollare i vari studi: forse c'è qualche considerazione interessante che si potrebbe fare sulla quantizzazione
> [!PDF|note] [[Survey_Splitting.pdf#page=10&selection=220,45,224,44&color=note|Survey_Splitting, p.10]]
> To the best of our knowledge, the papers in [26] and [90] were the first to propose altering existing DNN architectures to design relatively small bottlenecks at early layers in DNN models, instead of introducing compression techniques (e.g., quantization, autoencoder) to the models, so that communication delay (cost) and total inference time can be further reduced.

DOPO QUESTA PARTE VENGONO ANALIZZATE LE VARIE TECNICHE DI ADDESTRAMENTO QUANDO INTRODOTTO BOTTLENECK: ANALIZZARE SE C'È BISOGNO DI ESPANDERE IL DISCORSO

## Early Exiting

ANALIZZARE MEGLIO SE C'È BISOGNO DI ANALIZZARE IL DISCORSO

> [!PDF|note] [[Survey_Splitting.pdf#page=16&selection=105,0,110,71&color=note|Survey_Splitting, p.16]]
> The core idea of EE, first proposed by Teerapittayanon et al. [140], is to circumvent the need to make DNN models smaller by introducing early exits in the DNN, where execution is terminated at the first exit achieving the desired confidence on the input sample.

> [!PDF|note] [[Survey_Splitting.pdf#page=16&selection=122,10,124,93&color=note|Survey_Splitting, p.16]]
> Note that all the exits are executed until the desired confidence is reached; that is, the computational complexity up to that point increases. Thus, the classifiers added to the DNN model need to be simple; that is, they need to have fewer layers than the layers after the branches

> [!PDF|note] [[Survey_Splitting.pdf#page=16&selection=125,82,130,98&color=note|Survey_Splitting, p.16]]
> Teerapittayanon et al. [141] also apply this idea to mobile-edge-cloud computing systems; the smallest neural model is allocated to the mobile device, and if that model’s confidence for the input is not large enough, the intermediate output is forwarded to the edge server, where inference will continue using a mid-sized neural model with another exit. If the output still does not reach the target confidence, the intermediate layer’s output is forwarded to the cloud, which executes the largest neural model

> [!PDF|note] [[Survey_Splitting.pdf#page=18&selection=39,17,50,8&color=note|Survey_Splitting, p.18]]
> Using the concept of EE, Lo et al. [87] propose two different methods: (1) authentic operation and (2) dynamic network sizing. The first approach is used to determine whether the model input is transferred to the edge server, and the latter dynamically adjusts the number of layers to be used as an auxiliary neural model deployed on the mobile device for efficient usage of communication channels in EC systems.

VEDERE STUDI AFFINI SE SERVE ESPANDERE UN PO' IL DISCORSO


## Challenges

### Bottleneck Design
> [!PDF|note] [[Survey_Splitting.pdf#page=21&selection=70,22,73,69&color=note|Survey_Splitting, p.21]]
> As suggested in [96], important metrics include (1) bottleneck data size (or compression rate), (2) complexity of the head model executed on the mobile device, and (3) resulting model accuracy. As a principle, the smaller the bottleneck representation is, the lower the communication cost between the mobile device and edge server will be.

> [!PDF|note] [[Survey_Splitting.pdf#page=21&selection=81,33,89,50&color=note|Survey_Splitting, p.21]]
>  since mobile devices often have limited computing resources and may have other constraints such as energy consumption due to their battery capacities, SC should aim at minimizing their computational load by making head models as lightweight as possible. For instance, designing a small bottleneck at a very early stage of the DNN model enables a reduction in the computational complexity of the head model 

> [!PDF|note] [[Survey_Splitting.pdf#page=22&selection=11,0,14,31&color=note|Survey_Splitting, p.22]]
> In general, it is challenging to optimize bottleneck design and placement with respect to all three different metrics, and existing studies empirically design the bottlenecks and determine the placements. Thus, theoretical discussion on bottleneck design and placement should be an interesting research topic for future work.


### Expanding Domains
> [!PDF|note] [[Survey_Splitting.pdf#page=22&selection=33,0,48,14&color=note|Survey_Splitting, p.22]]
> The application domains of SC and (in minor part) EE remain primarily focused on image classification. This focus may be explained by the size of the input, which makes compression a relevant problem in many settings and the complexity of the models and tasks. However, there are many other unexplored domains from which SC would benefit.


### Information-Theoretic Perspective
> [!PDF|note] [[Survey_Splitting.pdf#page=22&selection=74,70,98,17&color=note|Survey_Splitting, p.22]]
> A possible approach to justify SC and EE can be found in the study of information bottlenecks (IBs), which were introduced in [142] as a compression technique in which a random variable X is compressed while preserving relevant information about another random variable Y. The IB method has been applied in [143] to quantify mutual information between the network layers and derive an information theory limit on DNN efficiency




CONTROLLARE LA SURVEY E VEDERE SE CI SONO ARTICOLI O PAPER CON APPROCCIO SIMILE A QUELLO USATO FINO AD ORA


# [[Neurosurgeon.pdf#page=1&selection=0,0,0,12|Neurosurgeon]]


> [!PDF|note] [[Neurosurgeon.pdf#page=2&selection=49,0,53,17&color=note|Neurosurgeon, p.2]]
> 2. At what point is the cost of transferring speech and image data over the wireless network too high to justify cloud processing?

> [!PDF|important] [[Neurosurgeon.pdf#page=2&selection=141,0,153,39&color=important|Neurosurgeon, p.2]]
> Neurosurgeon runtime system and layer performance prediction models – We develop a set of models to predict the latency and power consumption of a DNN layer based on its type and configuration, and create Neurosurgeon, a system to intelligently partition DNN computation between the mobile and cloud. 

## Status Quot
> [!PDF|note] [[Neurosurgeon.pdf#page=4&selection=240,13,243,37&color=note|Neurosurgeon, p.4]]
> loud processing has a significant computational advantage over mobile processing, but it does not always translate to end-to-end latency/energy advantage due to the dominating data transfer overhead

## Neurosurgeon
> [!PDF|note] [[Neurosurgeon.pdf#page=7&selection=376,0,381,17&color=note|Neurosurgeon, p.7]]
> The best partition point for a DNN architecture depends on the DNN’s topology, which manifests itself in the computation and data size variations of each layer. In addition, dynamic factors such as state of the wireless network and datacenter load affect the best partition point even for the same DNN architecture.

> [!PDF|note] [[Neurosurgeon.pdf#page=7&selection=408,0,412,37&color=note|Neurosurgeon, p.7]]
> Neurosurgeon profiles the mobile device and the server to generate performance prediction models for the spectrum of DNN layer types.  per-application profiling is not needed. This set of prediction models are stored on the mobile device and later used to predict the latency and energy cost of each layer 

### Performance Prediction Model
> [!PDF|note] [[Neurosurgeon.pdf#page=8&selection=207,0,217,18&color=note|Neurosurgeon, p.8]]
> Neurosurgeon models the per-layer latency and the energy consumption of arbitrary neural network architecture. This approach allows Neurosurgeon to estimate the latency and energy consumption of a DNN’s constituent layers without executing the DNN.

> [!PDF|note] [[Neurosurgeon.pdf#page=8&selection=218,0,225,10&color=note|Neurosurgeon, p.8]]
> We observe that for each layer type, there is a large latency variation across layer configurations. Thus, to construct the prediction model for each layer type, we vary the configurable parameters of the layer and measure the latency and power consumption for each configuration. Using these profiles, we establish a regression model for each layer type to predict the latency and power of the layer based on its con- figuration.

> [!PDF|note] [[Neurosurgeon.pdf#page=8&selection=228,0,232,33&color=note|Neurosurgeon, p.8]]
> Based on the layer type, we use either a logarithmic or linear function as the regression function. The logarithmic-based regression is used to model the performance plateau as the computation requirement of the layer approaches the limit of the available hardware resources.

> [!PDF|note] [[Neurosurgeon.pdf#page=8&selection=238,31,254,53&color=note|Neurosurgeon, p.8]]
> The regression model for convolution layer is based on two variables: the number of features in the input feature maps, and (f ilter size/stride)2 × (# of f ilters), which represents the amount of computation applied to each pixel in the input feature maps.

> [!PDF|note] [[Neurosurgeon.pdf#page=9&selection=232,23,237,33&color=note|Neurosurgeon, p.9]]
> he candidate points are after each layer. Lines 16 and 18 evaluate the performance when partitioning at each candidate point and select the point for either best end-to-end latency or best mobile energy consumption. Because of the simplicity of the regression models, this evaluation is lightweight and efficient


L'uso di un modello lineare/quadratico potrebbe essere interessante: mappare i flops per tipo di livello sull'output.


# [[Auto_Tuning.pdf#page=1&selection=0,0,3,0|Auto-Tuning Neural Network Quantization Framework for Collaborative Inference Between the Cloud and Edge]]

Non c'è un'effettiva ricerca dei livelli migliori da quantizzare, sono quantizzati tutti quelli sel device a prescindere.
> [!PDF|note] [[Auto_Tuning.pdf#page=2&selection=57,23,61,33&color=note|Auto_Tuning, p.2]]
> In the time of inference, the first part of the network is quantized and executed on the edge devices, and the second part of the network is executed in the cloud servers. On the edge, we use quantized neural network to reduce storage and computation. In the cloud, we use original full-precision network to achieve high accuracy.

La ricerca è fatta in modo statico, non viene cercato un livello ottimale.
> [!PDF|note] [[Auto_Tuning.pdf#page=3&selection=19,2,22,14&color=note|Auto_Tuning, p.3]]
> We analyze the structures of deep neural networks and show which layers are reasonable partition points. Based on the analysis, we could generate candidate layers as partition points of a specific neural network

# [[Deep_Feature_Compression.pdf#page=1&selection=0,0,0,59|DEEP FEATURE COMPRESSION FOR COLLABORATIVE OBJECT DETECTION]]

> [!PDF|note] [[Deep_Feature_Compression.pdf#page=2&selection=389,0,392,20&color=note|Deep_Feature_Compression, p.2]]
> A more efficient approach would be to compress the data prior to upload to the cloud. To achieve this, we could quantize the data, say to 8 bits per sample, then encode the quantized data losslessl

Questo può essere significativo per dire che nei modelli Yolo ci sono dei colli di bottiglia naturali
> [!PDF|note] [[Deep_Feature_Compression.pdf#page=4&selection=15,0,20,29&color=note|Deep_Feature_Compression, p.4]]
> As seen in the figure, when the split point is close to the input (e.g. max 3, conv 6 or conv 10 layers), the data volume is too large, and even with lossless compression of feature data, it is more efficient to simply upload input images to the cloud. But as we move down the network, it becomes more advantageous to upload feature data.

Il problema rispetto al mio caso è che questo approccio è usato solo come mezzo di compressione dei dati, non tenendo in conto il beneficio che ne deriva in termini di calcolo.
> [!PDF|note] [[Deep_Feature_Compression.pdf#page=4&selection=2,0,8,7&color=note|Deep_Feature_Compression, p.4]]
> We first test the impact of lossless compression (after the Q-layer) on accuracy. As is common with multi-class object detectors [18], we use mean Average Precision (mAP) as a measure of accuracy, and look at its variation with 8-bit, 10bit and 12-bit quantization in the Q-layer. The compression of feature data is quantified using average Kbits per image (KBPI).


# [[JointDNN.pdf#page=1&selection=2,0,2,8|JointDNN]]

> [!PDF|note] [[JointDNN.pdf#page=1&selection=54,57,56,39&color=note|JointDNN, p.1]]
> On the other side, mobile-device are being equipped with more powerful general-purpose CPUs and GPUs.

> [!PDF|note] [[JointDNN.pdf#page=1&selection=95,0,106,12&color=note|JointDNN, p.1]]
> Despite the recent improvements of the mobile devices mentioned earlier, the computational power of mobile devices is still significantly weaker than the cloud ones. Therefore, the mobile-only approach can cause large inference latency and failure in meeting QoS. Moreover, embedded devices undergo major energy consumption constraints due to battery limits. On the other hand, cloud-only suffers communication overhead for uploading the raw data and downloading the outputs. Moreover, slowdowns caused by service congestion, subscription costs, and network dependency should be considered as downsides of this approach [11].

> [!PDF|note] [[JointDNN.pdf#page=1&selection=112,0,116,47&color=note|JointDNN, p.1]]
> However, there is a trend of applications requiring adaptive learning in online environments, such as self-driving cars and security drones [12] [13]. Model parameters in these smart devices are constantly being changed based on their continuous interaction with their environment. 

> [!PDF|note] [[JointDNN.pdf#page=2&selection=2,0,7,47&color=note|JointDNN, p.2]]
> The main difference of collaborative training and cloudonly training is that the data transferred in the cloud-only approach is the input data and model parameters but in the collaborative approach, it is layer(s)’s output and a portion of model parameters. Therefore, the amount of data communicated can be potentially decreased [14].

> [!PDF|important] [[JointDNN.pdf#page=2&selection=42,0,48,56&color=important|JointDNN, p.2]]
> In this work, we are investigating the inference and training of DNNs in a joint platform of mobile and cloud as an alternative to the current single-platform methods

> [!PDF|note] [[JointDNN.pdf#page=2&selection=49,25,59,56&color=note|JointDNN, p.2]]
> Considering DNN architectures as an ordered sequence of layers, and the possibility of computation of every layer either on mobile or cloud, we can model the DNN structure as a Directed Acyclic Graph (DAG). The parameters of our real-time adaptive model are dependent on the following factors: mobile/cloud hardware and software resources, battery capacity, network specifications, and QoS. Based on this modeling, we show that the problem of finding the optimal computation schedule for different scenarios, i.e. best performance or energy consumption, can be reduced to the polynomial-time shortest path problem.

> [!PDF|note] [[JointDNN.pdf#page=2&selection=71,11,79,39&color=note|JointDNN, p.2]]
> his sequence suggests the computation of the first few layers on the mobile device to avoid excessive communication cost of uploading large raw input data. On the other hand, the growth of the layer output size from input to output in generative models which are used for synthesizing new data, implies the possibility of uploading a small input vector to the cloud and later downloading one of the last layers and performing the rest of computations on the mobile device for better efficiency.

> [!PDF|important] [[JointDNN.pdf#page=2&selection=100,0,106,9&color=important|JointDNN, p.2]]
> As we will see in Section 4, the communication between the mobile and cloud is the main bottleneck for both performance and energy in the collaborative approach. We investigated the specific characteristics of CNN layer outputs and introduced a loss-less compression approach to reduce the communication costs while preserving the model accuracy.

## Problem

> [!PDF|note] [[JointDNN.pdf#page=3&selection=30,0,39,10&color=note|JointDNN, p.3]]
> Statistical Modeling: In this method, a regression model over the configurable parameters of operators (e.g. filter size in the convolution) can be used to estimate the associated latency and energy. This method is prone to large errors because of the inter-layer optimizations performed by DNN software packages. Therefore, it is necessary to consider the execution of several consecutive operators grouped during profiling. 

> [!PDF|yellow] [[JointDNN.pdf#page=3&selection=42,0,54,56&color=yellow|JointDNN, p.3]]
> In order to illustrate this issue, we designed two experiments with 25 consecutive convolutions on NVIDIA PascalTM GPU using cuDNN R© library [18]. In the first experiment, we measure the latency of each convolution operator separately and set the total latency as the sum of them.

> [!PDF|note] [[JointDNN.pdf#page=3&selection=89,0,96,11&color=note|JointDNN, p.3]]
> Application-specific Profiling: In this method, the DNN architecture of the application being used is profiled in run-time. The number of applications in a mobile device using neural networks is generally limited. In conclusion, this method is more feasible, promising higher accuracy estimations

> [!PDF|note] [[JointDNN.pdf#page=3&selection=70,0,77,27&color=note|JointDNN, p.3]]
> Analytical Modeling: To derive analytical formulations for estimating the latency and energy consumption, it is required to obtain the exact hardware and software speci- fications. However, the state-of-the-art in latency modeling of DNNs [19] fails to estimate layer-level delay within an acceptable error bound, 

### Graph Model

Stanno assumento esecuzione prettamente sequenziale, senza la presenza di branch.
> [!PDF|note] [[JointDNN.pdf#page=3&selection=108,0,111,60&color=note|JointDNN, p.3]]
> First, we assume that a DNN is presented by a sequence of distinct layers with a linear topology as depicted in Figure 4. Layers are executed sequentially, with output data generated by one layer feeds into the input of the next one.


Questo è interessante: nel mio caso ho assunto soltanto il costo di upload dal server k al server k+1, senza considerare anche il costo di download.
> [!PDF|note] [[JointDNN.pdf#page=3&selection=214,0,230,2&color=note|JointDNN, p.3]]
> Downloading the Input Data cost of the kth layer is the cost of downloading output data of the (k-1)th layer to the mobile (DODk). 

> [!PDF|yellow] [[JointDNN.pdf#page=3&selection=241,16,284,14&color=yellow|JointDNN, p.3]]
> Node Ci:j represents that the layers i to j are computed on the cloud server, while node Mi:j represents that the layers i to j are computed on the mobile device.

> [!PDF|note] [[JointDNN.pdf#page=4&selection=321,0,329,50&color=note|JointDNN, p.4]]
> However, the problem of the shortest path subjected to constraints is NP-Complete [22]. For instance, assuming our standard graph is constructed for energy and we need to find the shortest path subject to the constraint of the total latency of that path is less than a time deadline (QoS). However, there is an approximation solution to this problem, ”LARAC” algorithm [23], the nature of our application does not require to solve this optimization problem frequently, therefore, we aim to obtain the optimal solution. 

![[JointDNN.pdf#page=6&rect=309,302,592,708&color=note|JointDNN, p.6|500]]

### Modellazione del consumo energetico di trasmissione

![[JointDNN.pdf#page=7&rect=308,305,574,494&color=note|JointDNN, p.7]]

![[JointDNN.pdf#page=7&rect=325,672,552,756&color=note|JointDNN, p.7]]

Nello studio viene anche analizzata la possibilità di usare compressione prima di trasferire l'output al fine di limitare l'impatto della trasmissione sul consumo e sulla latenza.


# [[Distributed_inference_Acceleration.pdf#page=1&selection=0,0,1,44|Distributed Inference Acceleration with Adaptive DNN Partitioning and Offloading]]

Viene assunto che tutti i nodi conoscano tutto il modello
> [!PDF|note] [[Distributed_inference_Acceleration.pdf#page=2&selection=493,23,498,42&color=note|Distributed_inference_Acceleration, p.2]]
> he DNN model of d is pre-loaded at all nodes in the network.


> [!PDF|note] [[Distributed_inference_Acceleration.pdf#page=2&selection=602,0,604,31&color=note|Distributed_inference_Acceleration, p.2]]
> The computation time of a DNN depends on the type of its layers. This work considers multi-channel convolution, feedforward, and activation layers.


Viene presa in considerazione una coda molto semplice
![[Distributed_inference_Acceleration.pdf#page=3&rect=307,625,572,718&color=note|Distributed_inference_Acceleration, p.3]]

Non stiamo distribuendo la rete neurale in questo caso! Ogni nodo può eseguire l'intera computazione e quello che stiamo facendo è vedere come piazzare la lista dei vari task
![[Distributed_inference_Acceleration.pdf#page=3&rect=308,451,570,607&color=note|Distributed_inference_Acceleration, p.3]]

Viene fatta anche un partizionamento della DNN.



FINIRE DI VEDERE MA NON MI SEMBRA CI SIANO COSE MOLTO SIGNIFICATIVE