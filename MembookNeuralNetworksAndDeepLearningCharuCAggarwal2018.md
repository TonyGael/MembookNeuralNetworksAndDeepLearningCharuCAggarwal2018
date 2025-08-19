***
# Redes Neuronales y Aprendizaje Profundo. Membook de conceptos

*Por Tony Gael - Agosto 2025*

*Basado en el libro: Neural Networks And Deep Learning de Charu C. Aggarwall de 2018.*

***
# Parte I
***

# Redes Neuronales Artificiales (RNA)

Las Redes Neuronales Artificiales (RNA), en el contexto de los conceptos fundamentales, son técnicas populares de aprendizaje automático que **simulan el mecanismo de aprendizaje en organismos biológicos**. El sistema nervioso humano contiene células llamadas **neuronas**, interconectadas por axones y dendritas, con regiones de conexión conocidas como **sinapsis**. La fuerza de estas conexiones sinápticas cambia en respuesta a estímulos externos, lo que constituye la base del aprendizaje en seres vivos.

Esta simulación se refleja en las RNA, que utilizan **unidades de cómputo, también llamadas neuronas**, interconectadas mediante **pesos** que cumplen la misma función que la fuerza de las conexiones sinápticas.

A continuación, se detallan los conceptos fundamentales de las RNA:

*   **Fundamento Biológico e Inspiración**:
    *   Las RNA se inspiran en la estructura del sistema nervioso humano, donde las neuronas se conectan a través de sinapsis.
    *   Los pesos en las RNA son análogos a la fuerza de las conexiones sinápticas.
    *   Aunque la comparación biológica es a menudo criticada como una "caricatura pobre" del cerebro humano, los principios de la neurociencia han sido útiles en el diseño de arquitecturas de redes neuronales.
    *   Ejemplos incluyen las redes neuronales convolucionales (CNN), inspiradas en los experimentos de Hubel y Wiesel sobre la corteza visual del gato.

*   **Unidades Computacionales y Pesos**:
    *   Cada unidad computacional (neurona) recibe entradas que son escaladas por un peso.
    *   La red calcula una función de las entradas propagando valores desde las neuronas de entrada a las de salida, usando los pesos como parámetros intermedios.
    *   El **aprendizaje** en las RNA ocurre **cambiando estos pesos**. Similar a los estímulos externos en organismos biológicos, los datos de entrenamiento (pares entrada-salida) proporcionan la información para ajustar los pesos. Los errores de predicción actúan como "retroalimentación desagradable", llevando al ajuste de los pesos para reducir el error en futuras iteraciones.

*   **El Perceptrón: La Arquitectura Más Sencilla**:
    *   El perceptrón es la red neuronal más simple, con una sola capa de entrada y un nodo de salida.
    *   La capa de entrada transmite las características con pesos a un nodo de salida, donde se calcula una función lineal ($W \cdot X$).
    *   La **función `sign`** se aplica posteriormente para predecir una variable de clase binaria (+1 o -1), sirviendo como **función de activación**.
    *   Un **sesgo (bias)** puede ser incorporado como el peso de una neurona que siempre transmite un valor de 1, capturando una parte invariante de la predicción.
    *   El algoritmo del perceptrón fue propuesto heurísticamente para minimizar errores de clasificación, con garantías de convergencia solo en casos de **datos linealmente separables**. Para datos no linealmente separables, el perceptrón tiende a rendir mal.

*   **Funciones de Pérdida (Loss Functions)**:
    *   La función de pérdida es crucial para definir las salidas de manera sensible a la aplicación.
    *   Para el perceptrón, el objetivo heurístico era minimizar el número de clasificaciones erróneas, que puede verse como una forma de **función de pérdida 0/1**. Sin embargo, esta función no es diferenciable.
    *   El **criterio del perceptrón**, $L_i = \max\{-y_i(W \cdot X_i), 0\}$, es una función de pérdida suave y subrogada que fue "ingeniería inversa" para explicar las actualizaciones del perceptrón.
    *   Otros ejemplos de funciones de pérdida incluyen:
        *   **Pérdida al cuadrado** ($(y - \hat{y})^2$) para regresión con salidas numéricas.
        *   **Pérdida de bisagra (hinge loss)** ($L = \max\{0, 1 - y \cdot \hat{y}\}$) utilizada en máquinas de vectores de soporte (SVM). El perceptrón se relaciona con el SVM lineal.
        *   **Pérdida de regresión logística** ($L = \log(1 + \exp(-y \cdot \hat{y}))$) para objetivos binarios.
        *   **Pérdida de entropía cruzada** ($L = -\log(\hat{y}_r)$) para objetivos categóricos (multiclase), a menudo con activación softmax. Generalmente, la entropía cruzada es más fácil de optimizar que la pérdida al cuadrado.

*   **Funciones de Activación**:
    *   La elección de la función de activación es una parte crítica del diseño de redes neuronales.
    *   Permiten **mapeos no lineales de los datos**, lo que es crucial para que las redes multicapa ganen poder de modelado. Si solo se usan activaciones lineales, una red multicapa no es más potente que una red lineal de una sola capa.
    *   Tipos comunes incluyen:
        *   **Identidad (Lineal)**: $\Phi(v) = v$. Se usa a menudo en el nodo de salida para objetivos de valor real.
        *   **Signo**: $\Phi(v) = \text{sign}(v)$. Utilizada por el perceptrón para predicción binaria, pero su no diferenciabilidad impide su uso en la función de pérdida durante el entrenamiento.
        *   **Sigmoide**: $\Phi(v) = \frac{1}{1 + e^{-v}}$. Salida en $(0, 1)$, útil para interpretaciones probabilísticas.
        *   **Tangente Hiperbólica (Tanh)**: $\Phi(v) = \frac{e^{2v} - 1}{e^{2v} + 1}$. Salida en $[-1, 1]$, preferible a la sigmoide cuando se desean salidas positivas y negativas, y tiene un gradiente más grande.
        *   **Unidad Lineal Rectificada (ReLU)**: $\Phi(v) = \max\{v, 0\}$. Ha reemplazado en gran medida a la sigmoide y tanh en redes modernas debido a la facilidad de entrenamiento. Es menos propensa al problema del gradiente desvaneciente.
        *   **Tanh dura (hard tanh)**: $\Phi(v) = \max\{\min[v, 1], -1\}$.
    *   Las funciones de activación no lineales son la **clave para aumentar el poder de modelado de una red**.

*   **Redes Neuronales Multicapa (Feed-Forward)**:
    *   Contienen más de una capa computacional. Las capas adicionales entre la entrada y la salida se denominan **capas ocultas**, ya que sus cómputos no son directamente visibles para el usuario.
    *   La arquitectura **feed-forward** implica que las capas sucesivas se alimentan entre sí en dirección hacia adelante (de entrada a salida).
    *   Los pesos de las conexiones entre capas se representan mediante **matrices de pesos**.
    *   La salida de una capa sirve como entrada para la siguiente, transformándose recursivamente mediante funciones de activación.
    *   La capacidad de las redes neuronales para **aproximar cualquier función "razonable"** (universal function approximators) se logra con una sola capa oculta de unidades no lineales, aunque esto puede requerir un número muy grande de unidades y parámetros.

*   **Red como Grafo Computacional y Backpropagation**:
    *   Las redes neuronales pueden verse como un **grafo computacional** que realiza composiciones de funciones más simples para crear funciones complejas.
    *   El algoritmo de **backpropagation (retropropagación)** es el método principal para entrenar redes neuronales multicapa.
    *   Utiliza la **regla de la cadena del cálculo diferencial** y la **programación dinámica** para calcular eficientemente los gradientes del error con respecto a los pesos en todas las capas.
    *   Consta de dos fases:
        *   **Fase hacia adelante (Forward phase)**: Los datos de entrada se propagan a través de la red para computar la salida y las derivadas locales.
        *   **Fase hacia atrás (Backward phase)**: Los gradientes de la función de pérdida se aprenden en dirección inversa (desde la salida), y se utilizan para actualizar los pesos.

*   **Problemas Prácticos en el Entrenamiento**:
    *   **Sobreajuste (Overfitting)**: El modelo rinde muy bien en los datos de entrenamiento pero pobremente en datos no vistos. Ocurre cuando hay demasiados parámetros libres en comparación con el tamaño de los datos de entrenamiento, llevando a la memorización de ruidos.
        *   **Estrategias de mitigación**: **Regularización** (como la penalización Tikhonov/decaimiento de peso), **diseño de arquitectura** y **compartición de parámetros** (ej. CNNs), **parada temprana**, **preferir profundidad sobre anchura** (menos unidades por capa), y **métodos de ensamblaje** (ej. Dropout, Dropconnect).
    *   **Problemas de Gradiente Desvaneciente y Explosivo (Vanishing and Exploding Gradient Problems)**: Los gradientes en capas anteriores pueden volverse insignificantes (desvanecientes) o extremadamente grandes (explosivos) en redes muy profundas, dificultando el entrenamiento.
        *   **Soluciones**: Uso de activaciones **ReLU** (derivada constante de 1 para valores positivos), tasas de aprendizaje adaptativas, y normalización por lotes (batch normalization).
    *   **Dificultades en la Convergencia**: Las redes muy profundas pueden tardar mucho en converger.
    *   **Óptimos Locales y Espurios**: La función de optimización no lineal tiene muchos óptimos locales. La **pre-entrenamiento** puede ayudar a encontrar mejores puntos de inicialización y evitar óptimos espurios.
    *   **Desafíos Computacionales**: El entrenamiento puede ser muy costoso en tiempo. Las **GPUs (Unidades de Procesamiento Gráfico)** han acelerado significativamente las operaciones de las redes neuronales.

*   **Poder de la Composición de Funciones y Profundidad**:
    *   El poder del "deep learning" radica en la **composición repetida de múltiples funciones no lineales**, lo que aumenta significativamente el poder expresivo de la red y **reduce el espacio de parámetros** necesario para el aprendizaje.
    *   Las **redes más profundas** pueden aprender características jerárquicas y regularidades repetidas en los datos con **menos datos** que las redes poco profundas. Por ejemplo, las CNNs aprenden líneas y bordes en capas tempranas y formas complejas como caras en capas posteriores.

*   **Relación con el Aprendizaje Automático Tradicional**:
    *   Las unidades más básicas de las RNA se inspiran en algoritmos tradicionales de aprendizaje automático como la regresión por mínimos cuadrados y la regresión logística.
    *   Los modelos básicos de aprendizaje automático, como la regresión lineal, la clasificación, las SVM y la regresión logística, pueden ser simulados con **redes neuronales "shallow" (poco profundas)**, con no más de una o dos capas.
    *   La verdadera potencia de las RNA se desata al **combinar múltiples unidades** y entrenar sus pesos conjuntamente, lo que les permite aprender funciones más complicadas.

En resumen, las Redes Neuronales Artificiales son modelos computacionales que, inspirados biológicamente, aprenden a través del ajuste iterativo de pesos, optimizando una función de pérdida. Su poder reside en la **composición de funciones no lineales en múltiples capas (profundidad)**, lo que les permite aprender representaciones de características jerárquicas y superar las limitaciones de los modelos lineales, aunque enfrentan desafíos prácticos como el sobreajuste y los problemas de gradiente, que se abordan con técnicas de diseño y entrenamiento avanzadas.


***
# Parte II
***

# Aprendizaje automático con neuronas superficiales (Machine Learning with Shallow Neural)

Las redes neuronales poco profundas son modelos parametrizados que se aprenden utilizando métodos de optimización continua, como el descenso de gradiente. Una variedad de métodos centrados en la optimización del aprendizaje automático convencional pueden ser representados con arquitecturas de redes neuronales muy simples, conteniendo una o dos capas. De hecho, estas redes neuronales pueden verse como versiones más potentes de modelos simples, logrando su capacidad al combinar estos modelos básicos en una arquitectura neuronal comprensiva (es decir, un grafo computacional).

**Relación con el Aprendizaje Automático Convencional:**
*   **Paralelos y composición**: Es útil ver el diseño de una red profunda como una **composición de unidades básicas** utilizadas en el aprendizaje automático tradicional. Esto permite apreciar cómo el aprendizaje automático tradicional difiere de las redes neuronales y cuándo las redes neuronales pueden ofrecer un mejor rendimiento.
*   **Ventaja de los datos**: Las redes neuronales tienen una **ventaja a medida que aumenta la cantidad de datos** disponibles, ya que retienen la flexibilidad para modelar funciones más complejas con la adición de neuronas al grafo computacional (ver Figura 2.1).
*   **Datos limitados y optimización**: Sin embargo, las arquitecturas neuronales complejas o profundas a menudo son **excesivas cuando hay poca cantidad de datos**. En estos entornos con pocos datos, es más fácil optimizar los modelos de aprendizaje automático tradicionales, ya que son más interpretables.
*   **Establecimiento de relaciones**: Existen **relaciones estrechas entre las primeras redes neuronales** (como el perceptrón y el aprendizaje de Widrow-Hoff) **y los modelos tradicionales** (como las máquinas de vectores de soporte y el discriminante de Fisher). Estas relaciones a menudo pasaron desapercibidas durante años debido a que los modelos fueron propuestos independientemente por diferentes comunidades.

**Modelos Básicos como Redes Neuronales Poco Profundas:**
El capítulo explora cómo los modelos fundamentales de aprendizaje automático pueden entenderse como unidades computacionales elementales en las redes neuronales. Se discuten dos clases principales de modelos:

1.  **Modelos Supervisados**:
    *   **Regresión de mínimos cuadrados**: El objetivo es minimizar el error cuadrático total. Su arquitectura es similar al perceptrón, pero con la función de activación de identidad y la pérdida cuadrática. El **aprendizaje de Widrow-Hoff** es una aplicación directa de la regresión de mínimos cuadrados a objetivos binarios, conocido también como clasificación de mínimos cuadrados o método lineal de mínimos cuadrados. El **discriminante de Fisher** para objetivos binarios es idéntico al aprendizaje de Widrow-Hoff. Una desventaja de Widrow-Hoff/Fisher es que penaliza las clasificaciones "demasiado correctas".
    *   **Perceptrón**: Propuesto históricamente con actualizaciones de descenso de gradiente antes de que se propusiera una función de pérdida explícita. El perceptrón utiliza una función de activación de signo para predicciones discretas y su criterio de pérdida es $Li = max\{0, -yi(W \cdot Xi)\}$ (el criterio del perceptrón), que es una aproximación suave de su objetivo. Las actualizaciones del perceptrón se realizan solo para instancias mal clasificadas.
    *   **Regresión logística**: Es un modelo probabilístico que clasifica instancias en términos de probabilidades. Utiliza la función de activación sigmoide en la capa de salida y la **negativa log-verosimilitud** como función de pérdida. Sus actualizaciones de descenso de gradiente utilizan las probabilidades de los errores. Una arquitectura alternativa utiliza la activación de identidad y la función de pérdida $Li = log(1 + exp(-yi \cdot \hat{y}i))$.
    *   **Máquinas de vectores de soporte (SVM)**: Su función de pérdida es el **"hinge-loss"** ($Li = max\{0, 1 - yi\hat{y}i\}$). A diferencia de Widrow-Hoff, las SVM no penalizan las predicciones "demasiado correctas". La SVM realiza actualizaciones para puntos mal clasificados o aquellos que están correctamente clasificados pero no con suficiente confianza. La **SVM L2-loss de Hinton** es una variación que repara la pérdida de Widrow-Hoff para evitar penalizar el sobre-rendimiento.
    *   **Resumen de funciones de pérdida**: Los modelos mencionados (Perceptrón, Widrow-Hoff/Fisher, Regresión Logística, SVM Hinge y SVM L2-Loss de Hinton) pueden ser vistos como variaciones del perceptrón, con sus funciones de pérdida detalladas en una tabla. Las actualizaciones de descenso de gradiente estocástico son comunes en el aprendizaje automático tradicional y en las redes neuronales, independientemente de la arquitectura neuronal.

2.  **Modelos Multiclase**:
    *   **Perceptrón Multiclase**: Aprende *k* separadores lineales simultáneamente, penalizando las predicciones incorrectas y las más mal clasificadas. Solo se actualizan dos separadores a la vez.
    *   **SVM de Weston-Watkins**: Generaliza la SVM binaria. Actualiza el separador de cualquier clase que sea predicha más favorablemente que la clase verdadera, y también cuando una clase incorrecta se acerca "incómodamente" a la clase verdadera, basándose en el concepto de margen. La pérdida se calcula como una suma de las contribuciones de las clases incorrectas.
    *   **Regresión Logística Multinomial (Clasificador Softmax)**: Generalización multiclase de la regresión logística, utilizando pérdida de negativa log-verosimilitud y la función de activación softmax para estimar probabilidades de clase. Actualiza todos los *k* separadores para cada instancia de entrenamiento.
    *   **Softmax Jerárquico**: Técnica para mejorar la eficiencia en problemas con un número extremadamente grande de clases (por ejemplo, más de 100,000) mediante la descomposición jerárquica del problema de clasificación en una estructura de árbol binario.

**Interpretación y Selección de Características:**
*   Las redes neuronales, a menudo criticadas por su falta de interpretabilidad, pueden usar la retropropagación para determinar qué características contribuyen más a la clasificación de una instancia de prueba particular. Esto se logra calculando la magnitud absoluta de la derivada parcial de la salida de la clase ganadora con respecto a cada característica.

**Factorización de Matrices con Autoencoders (Aprendizaje No Supervisado)**:
*   **Principio Básico**: Un autoencoder es una arquitectura fundamental para el aprendizaje no supervisado, incluyendo la factorización de matrices, el análisis de componentes principales (PCA) y la reducción de dimensionalidad. Su idea básica es tener una capa de salida con la misma dimensionalidad que las entradas y tratar de reconstruir cada dimensión exactamente. Esto se logra a través de **capas ocultas "constreñidas"** (con menos unidades que la entrada), lo que fuerza al autoencoder a aprender una representación reducida de los datos.
*   **Encoder y Decoder**: La parte inicial de la arquitectura que reduce la dimensionalidad se llama **codificador (encoder)**, y la parte final que reconstruye desde el código se llama **decodificador (decoder)**.
*   **Conexión con SVD y PCA**: Un autoencoder con una sola capa oculta y activación lineal puede simular la **descomposición de valores singulares (SVD)** y el **análisis de componentes principales (PCA)**. Al atar los pesos entre el codificador y el decodificador ($W = V^T$), se puede simular exactamente SVD, asegurando que las columnas de $V$ sean ortonormales.
*   **Activaciones No Lineales**: La verdadera potencia de los autoencoders se logra al usar activaciones no lineales y múltiples capas. Por ejemplo, para matrices binarias, una función sigmoide en la capa final y pérdida de logaritmo negativo conduce a la **factorización logística de matrices**. Las activaciones no lineales también pueden usarse en las capas ocultas para simular la **factorización no negativa de matrices**.
*   **Autoencoders Profundos**: Las redes profundas con múltiples capas proporcionan una capacidad de representación extraordinaria y reducciones jerárquicas de datos. Permiten **reducciones de dimensionalidad no lineales** que son más potentes que PCA para datos distribuidos en colectores curvos (ver Figura 2.9, Figura 2.11). Además, pueden manejar fácilmente datos fuera de muestra.
*   **Detección de Anomalías (Outliers)**: Los autoencoders son útiles para la detección de anomalías. Los puntos anómalos son difíciles de codificar y decodificar sin perder información sustancial; por lo tanto, los errores de reconstrucción pueden usarse como puntuaciones de anomalías para filas, columnas o entradas individuales de una matriz.
*   **Representaciones Sobrecompletas (Sparse Feature Learning)**: Cuando la capa oculta tiene más unidades que la entrada (representaciones sobrecompletas), se pueden aprender **características dispersas (sparse features)** imponiendo restricciones de dispersión, como penalizaciones L1 o permitiendo que solo las r-activaciones principales sean distintas de cero.
*   **Otras Aplicaciones**: Se utilizan para el **denoising** (autoencoders de-noising) al añadir ruido a los datos de entrenamiento pero calculando la pérdida con respecto a los datos originales. También son la base de los **autoencoders variacionales** y las **redes generativas antagónicas (GANs)** para generar muestras de datos realistas o creaciones artísticas. También son útiles para la **incrustación multimodal** de datos heterogéneos en un espacio latente conjunto (ver Figura 2.12) y para **pre-entrenar** redes neuronales en tareas supervisadas.

**Sistemas de Recomendación (Word2vec y Graph Embeddings como Aplicaciones Específicas)**:
*   **Recomendación basada en factorización de matrices**: Los autoencoders se pueden adaptar para sistemas de recomendación con matrices de calificaciones incompletas. Se utiliza una arquitectura donde la entrada es un índice de fila codificado en "one-hot" (por ejemplo, un usuario), y la salida son las calificaciones para todos los ítems. La formación se realiza utilizando solo las entradas conocidas (calificaciones observadas), lo que se relaciona con el **muestreo negativo**. Las actualizaciones de descenso de gradiente para esta arquitectura son idénticas a las de la factorización de matrices en sistemas de recomendación.
*   **Word2vec**: Métodos para crear **incrustaciones de palabras (word embeddings)** que capturan el orden secuencial de las palabras.
    *   **CBOW (Continuous Bag-of-Words)**: Predice una palabra objetivo a partir de su contexto (ventana de palabras). Agrega las incrustaciones de las palabras del contexto para crear la representación de la capa oculta. Utiliza softmax en la capa de salida.
    *   **Skip-gram Model**: Predice las palabras del contexto a partir de una palabra objetivo. Es una versión invertida de CBOW.
    *   **Skip-gram con Muestreo Negativo (SGNS)**: Una alternativa eficiente al softmax jerárquico. En lugar de predecir cada palabra del contexto, predice la presencia o ausencia de pares palabra-contexto utilizando una capa de sigmoides (modelo Bernoulli) y muestreo negativo de contextos. **SGNS es matemáticamente una factorización logística de matrices**. El modelo skip-gram original es una **factorización de matrices multinomial**.
*   **Incrustaciones de Grafos (Graph Embeddings)**:
    *   Similar a word2vec, se busca incrustar nodos de un grafo en vectores de características que capturen las relaciones entre ellos.
    *   Se puede usar la factorización logística de matrices para esto, donde la matriz de adyacencia binaria del grafo ($B$) se modela con parámetros de una distribución Bernoulli obtenidos de $f(UV^T)$, donde $f(\cdot)$ es la función sigmoide.
    *   Se utiliza el muestreo negativo para manejar la escasez de los grafos (muchos 0s en la matriz de adyacencia).
    *   Los modelos **DeepWalk** y **node2vec** se consideran variantes de modelos multinomiales (como el skip-gram vanilla) con pasos de preprocesamiento especializados (caminatas aleatorias para generar afinidades entre nodos).

**Ventajas de la Modularidad de las Redes Neuronales**:
*   La **naturaleza modular** de las redes neuronales facilita la experimentación con modelos sofisticados. Cambiar de una arquitectura a otra (por ejemplo, de regresión lineal a regresión logística) puede ser tan simple como cambiar unas pocas líneas de código en el software de redes neuronales.
*   La retropropagación se encarga de los detalles de la optimización, lo que **protege al usuario de las complejidades** de los pasos matemáticos subyacentes.

En resumen, las redes neuronales poco profundas sirven como una **abstracción unificadora** para muchos algoritmos de aprendizaje automático tradicionales, y su diseño modular y la facilidad de experimentación (gracias a la retropropagación) ofrecen un camino natural hacia la **generalización a modelos no lineales más potentes** y la exploración de nuevas variaciones algorítmicas.

***

# Parte III

***

# Entrenamiento y Estabilización de Redes Neuronales Profundas

### Introducción y Contexto Histórico
El Capítulo 3 se enfoca en la **descripción expandida del procedimiento para entrenar redes neuronales con retropropagación** (backpropagation), presentando el algoritmo con mayor detalle e incluyendo aspectos de implementación. Además, aborda la **pre-procesamiento de características e inicialización**, procedimientos computacionales con descenso de gradiente, el efecto de la profundidad de la red en la estabilidad del entrenamiento, métodos para solucionar estos problemas, y temas de eficiencia como la compresión de modelos entrenados para despliegue en dispositivos móviles.

En los primeros años, los métodos para entrenar redes multicapa eran desconocidos, lo que llevó a Minsky y Papert  a argumentar fuertemente en contra de las redes neuronales. Esto hizo que las redes neuronales perdieran popularidad hasta la década de 1980. El **algoritmo de retropropagación (backpropagation)**, propuesto por Rumelhart et al. , fue el primer avance significativo que reavivó el interés en ellas. Sin embargo, surgieron desafíos computacionales, de estabilidad y de sobreajuste, lo que nuevamente llevó a que las redes neuronales cayeran en desuso.

A principios del siglo XXI, varios avances revivieron la popularidad de las redes neuronales, no todos algorítmicos. La **mayor disponibilidad de datos** y el **aumento del poder computacional** jugaron un papel principal. Cambios en el algoritmo básico de retropropagación y métodos inteligentes de inicialización, como el pre-entrenamiento, también contribuyeron. La reducción de los tiempos de ciclo de prueba, debido a mejoras en el hardware, facilitó la experimentación intensiva necesaria para ajustes algorítmicos. Así, el **aumento de datos, el poder computacional y la reducción del tiempo de experimentación** (para ajustes algorítmicos) fueron de la mano.

Una característica clave del algoritmo de retropropagación es su **inestabilidad ante cambios menores en la configuración algorítmica**, como el punto de inicialización, especialmente en redes muy profundas. La optimización de redes neuronales es un problema de optimización multivariable, donde los pesos de las conexiones son las variables. Los problemas multivariables a menudo enfrentan desafíos de estabilidad, ya que los pasos deben tomarse en la proporción "correcta" en cada dirección, lo cual es difícil en este dominio. Un gradiente solo proporciona una tasa de cambio sobre un horizonte infinitesimal, pero un paso real tiene una longitud finita, y los gradientes pueden cambiar drásticamente sobre esta longitud finita. Esto hace que la optimización sea más compleja de lo que parece a primera vista, aunque muchos problemas pueden evitarse adaptando los pasos de descenso de gradiente para ser más robustos a la superficie de optimización.

### El Algoritmo de Retropropagación (Backpropagation)
La retropropagación es un **algoritmo iterativo basado en la programación dinámica y la regla de la cadena del cálculo diferencial** para calcular las derivadas de manera eficiente.

*   **Abstracción de Grafo Computacional**: Una red neuronal se concibe como un grafo computacional, donde cada neurona es una unidad de computación. La capacidad de las redes para crear funciones de composición altamente optimizadas, junto con las activaciones no lineales entre capas, les da su poder expresivo. Calcular las derivadas de funciones de composición anidadas, especialmente en redes profundas, es extremadamente complejo, llevando a un número exponencial de términos si se hiciera directamente. De ahí la necesidad de un enfoque iterativo.
*   **Reglas de la Cadena**:
    *   **Univariada**: Para una composición simple `f(g(w))`, la derivada `∂f(g(w))/∂w` es `(∂f(g(w))/∂g(w)) · (∂g(w)/∂w)`. Cada término es un gradiente local que simplifica el cálculo.
    *   **Multivariable**: Cuando una unidad recibe entradas de múltiples unidades `g1(w), ..., gk(w)`, se usa la regla de la cadena multivariable. Esta es una generalización de la regla univariada.
*   **Lema de Agregación de Trayectorias**: Este lema establece que la derivada `∂o/∂w` se obtiene calculando el producto de los gradientes locales a lo largo de cada trayectoria `P` desde la variable de entrada `w` hasta el nodo de salida `o`, y luego sumando estos productos sobre todas las trayectorias. Sin embargo, el número de trayectorias en un grafo computacional aumenta exponencialmente con la profundidad, lo que hace que un cálculo explícito sea inviable (un algoritmo de tiempo exponencial). Por ejemplo, una red con 5 capas y 2 unidades por capa tiene 2^5 = 32 trayectorias.
*   **La Programación Dinámica al Rescate**: La retropropagación utiliza la programación dinámica para agregar eficientemente el producto de los gradientes locales a lo largo de las exponencialmente numerosas trayectorias en un grafo computacional. La actualización de programación dinámica es idéntica a la regla de la cadena multivariable, pero aplicada en un orden específico para minimizar los cálculos.

#### Vistas de Backpropagation
El algoritmo se puede instanciar usando diferentes tipos de variables en los nodos del grafo computacional:

*   **Variables Post-Activación**: Los nodos contienen las variables post-activación (variables ocultas de diferentes capas).
    *   **Fase Hacia Adelante (Forward Pass)**: Se utiliza para calcular los valores de cada capa oculta y la salida para una entrada dada, así como la función de pérdida.
    *   **Fase Hacia Atrás (Backward Pass)**: Calcula el gradiente de la función de pérdida con respecto a los pesos. Se inicializa el gradiente en el nodo de salida y se propaga hacia atrás utilizando la regla de la cadena multivariable.
*   **Variables Pre-Activación**: Los gradientes se calculan con respecto a los valores pre-activación de las variables ocultas. Esta es la forma más común en muchos libros de texto y tiene la ventaja de que el gradiente de activación está fuera de la sumatoria, lo que simplifica la computación y permite desacoplar la función de activación de la transformación lineal en las actualizaciones. Ambas variantes (pre- y post-activación) son **matemáticamente equivalentes**.

#### Actualizaciones para Diferentes Funciones de Activación
La formulación con variables pre-activación (Ecuación 3.18) facilita la derivación de actualizaciones específicas:
*   **Lineal**: `δ(hr, o) = ∑ h:hr⇒h w(hr,h)δ(h, o)`.
*   **Sigmoide**: `δ(hr, o) = hr(1 − hr) ∑ h:hr⇒h w(hr,h)δ(h, o)`.
*   **Tanh**: `δ(hr, o) = (1 − h2r) ∑ h:hr⇒h w(hr,h)δ(h, o)`.
*   **ReLU**: `δ(hr, o) = ∑ h:hr⇒h w(hr,h)δ(h, o)` si `0 < ahr`, `0` en otro caso.
*   **Hard Tanh**: `δ(hr, o) = ∑ h:hr⇒h w(hr,h)δ(h, o)` si `-1 < ahr < 1`, `0` en otro caso.

#### Caso Especial de Softmax
La activación Softmax es especial porque se calcula con respecto a múltiples entradas, no solo una. Cuando se usa en la capa de salida y se empareja con la pérdida de entropía cruzada, la derivada de la pérdida `L` con respecto a la entrada `vi` de Softmax se simplifica a `oi - yi` (donde `oi` es la probabilidad de salida y `yi` es la etiqueta codificada one-hot). Esto permite desacoplar la retropropagación de la activación Softmax del resto de la red.

#### Vista Desacoplada de Backpropagation Vector-Céntrica
En las implementaciones reales, las **operaciones lineales (multiplicación matricial) y las funciones de activación se desacoplan en "capas" separadas**. Las capas de activación realizan cálculos elemento por elemento, mientras que las capas lineales realizan cálculos de todo a todo. Esta vista simplifica enormemente los cálculos y es eficiente en hardware optimizado para matrices como las **GPUs**. Las ecuaciones de retropropagación se escriben como multiplicaciones matriciales (`g_i = W g_{i+1}` para capas lineales y `g_{i+1} = g_{i+2} \bigodot \Phi'(z_{i+1})` para capas de activación, donde `\bigodot` es multiplicación elemento a elemento), lo cual es beneficioso para la aceleración en GPUs.

#### Múltiples Nodos de Salida y Nodos Ocultos con Función de Pérdida
La retropropagación puede generalizarse a múltiples nodos de salida, sumando las contribuciones de las diferentes salidas a las derivadas de la pérdida. Además, si los nodos ocultos tienen funciones de pérdida asociadas (ej. para escasez o regularización), el algoritmo se modifica mínimamente: el flujo de gradiente hacia atrás agrega las contribuciones de todas las pérdidas asociadas a los nodos alcanzables.

#### Descenso de Gradiente Estocástico por Mini-Lotes (Mini-Batch Stochastic Gradient Descent)
Las actualizaciones de pesos se realizan comúnmente de manera punto-específica (descenso de gradiente estocástico) o por mini-lotes.
*   La función de pérdida total `L` es la suma de las pérdidas `L_i` de puntos de entrenamiento individuales.
*   El **descenso de gradiente tradicional** calcula la pérdida sobre *todos* los puntos simultáneamente para cada actualización, lo cual es inviable para grandes conjuntos de datos debido a los requisitos de memoria y cómputo.
*   El **descenso de gradiente estocástico (SGD)** actualiza los pesos utilizando el gradiente de un *solo* punto de entrenamiento a la vez. Es eficiente localmente, aunque cada actualización es una aproximación probabilística. A menudo se desempeña comparable o mejor en datos de prueba que el descenso de gradiente completo, actuando como regularización.
*   El **descenso de gradiente estocástico por mini-lotes** utiliza un subconjunto (mini-lote) de puntos de entrenamiento para cada actualización. Ofrece un **mejor equilibrio entre estabilidad, velocidad y requisitos de memoria**. El tamaño del mini-lote, a menudo una potencia de 2 (32, 64, 128, 256), se regula por la memoria disponible y más allá de unos cientos de puntos no mejora significativamente la precisión del gradiente.

#### Pesos Compartidos
Cuando los pesos se comparten entre diferentes nodos de la red (común en autoencoders, redes neuronales recurrentes, convolucionales), la retropropagación es matemáticamente sencilla: **se pretenden que los pesos son independientes, se calculan sus derivadas y luego se suman**.

#### Verificación de la Correctitud del Cálculo del Gradiente
La retropropagación es compleja, por lo que es útil verificar la correctitud de los gradientes numéricamente. Se perturba ligeramente un peso `w` (`w + ε`), se ejecuta el forward pass para obtener `L(w + ε)`, y se compara la derivada calculada por retropropagación (`Ge`) con la aproximación numérica `(L(w + ε) - L(w)) / ε` (`Ga`). La **relación `ρ = |Ge − Ga| / |Ge + Ga|` debe ser típicamente inferior a 10^-6** (o 10^-3 para funciones como ReLU con cambios bruscos).

### Configuración y Problemas de Inicialización
La selección de hiperparámetros, el pre-procesamiento de características y la inicialización son cruciales debido al gran espacio de parámetros de las redes neuronales.

*   **Ajuste de Hiperparámetros**: Son parámetros que regulan el diseño del modelo (ej. tasa de aprendizaje, fuerza de regularización). Se ajustan utilizando un **conjunto de validación**, no los datos de entrenamiento para evitar el sobreajuste.
    *   **Búsqueda en Cuadrícula (Grid Search)**: Prueba todas las combinaciones de valores seleccionados para cada hiperparámetro. Puede ser computacionalmente muy costosa, por lo que se usan cuadrículas más gruesas y luego más finas.
    *   **Muestreo Aleatorio**: Muestrea los hiperparámetros uniformemente dentro de un rango, a menudo en el **espacio logarítmico** para tasas de aprendizaje y regularización.
    *   **Terminación Temprana**: Para entrenamientos largos, se prueban los algoritmos por un número de épocas y se detienen las ejecuciones pobres o divergentes.
    *   **Optimización Bayesiana**: Un método matemáticamente justificado para elegir hiperparámetros, pero a menudo demasiado lento para redes neuronales a gran escala, aunque útil para redes más pequeñas.
*   **Pre-procesamiento de Características**:
    *   **Centrado en la Media y aditivo**: Restar la media de cada columna o sumar el valor absoluto de la entrada más negativa para asegurar valores no negativos.
    *   **Normalización de Características**:
        *   **Estandarización**: Dividir cada valor por su desviación estándar, a menudo combinada con el centrado en la media, asumiendo una distribución normal estándar (media cero, varianza unitaria).
        *   **Normalización Min-Max**: Escalar los datos a un rango (0, 1).
    *   **Blanqueamiento (Whitening)**: Rotar el sistema de ejes para crear un nuevo conjunto de características descorrelacionadas, cada una escalada a varianza unitaria. Típicamente se usa el Análisis de Componentes Principales (PCA). Esto da igual importancia a diferentes características y previene que las "grandes" dominen las activaciones y gradientes iniciales.
*   **Inicialización**: Es vital para la estabilidad del entrenamiento.
    *   **Valores aleatorios pequeños**: Generados a partir de una distribución Gaussiana con media cero y una desviación estándar pequeña (ej. 10^-2).
    *   **Escalamiento por número de entradas**: Inicializar cada peso de una distribución Gaussiana con desviación estándar `1/√r` (donde `r` es el número de entradas a la neurona) o uniformemente en `[-1/√r, 1/√r]`. Los sesgos se inicializan a cero.
    *   **Inicialización Xavier/Glorot**: Utiliza una distribución Gaussiana con desviación estándar `√(2/(rin + rout))` donde `rin` y `rout` son el fan-in y fan-out respectivamente.
    *   **Ruptura de Simetría**: Es crucial que los pesos no se inicialicen al mismo valor (ej. 0) para evitar que todas las actualizaciones se muevan al unísono y creen características idénticas.

### Problemas de Gradientes Desvanecidos y Explotados
Las redes neuronales profundas tienen problemas de estabilidad: los gradientes en capas anteriores se vuelven sucesivamente más débiles (desvanecidos) o más fuertes (explotados). Esto se debe a la multiplicación repetida de términos pequeños (en el caso de sigmoid, el gradiente máximo es 0.25) o grandes. Afecta desproporcionadamente a las capas, causando actualizaciones muy pequeñas o muy grandes, lo que ralentiza el progreso o causa inestabilidad.

*   **Comprensión Geométrica**: Estos problemas son inherentes a la optimización multivariable. En superficies de pérdida complejas (como cuencos elípticos), el descenso más pronunciado puede oscilar o serpentear, en lugar de apuntar directamente al óptimo, requiriendo muchas correcciones de rumbo. Los gradientes desvanecidos son una manifestación extrema de esto, requiriendo un número extremadamente grande de pequeñas actualizaciones.
*   **Elección de la Función de Activación (Solución Parcial)**:
    *   **Sigmoide y Tanh**: Son propensas al problema del gradiente desvanecido debido a sus pequeños gradientes (sigmoide no más de 0.25) y la saturación en valores absolutos grandes, lo que ralentiza el cambio de pesos.
    *   **ReLU y Hard Tanh**: Son más populares porque tienen un gradiente de 1 en ciertos intervalos, lo que reduce el problema del gradiente desvanecido, siempre que las unidades operen en esos intervalos. Son más rápidas de entrenar.
*   **Neuronas Muertas ("Brain Damage")**: Un problema con ReLU donde los gradientes son cero para valores de entrada negativos. Si una neurona ReLU siempre tiene una entrada negativa (ej. por una alta tasa de aprendizaje o inicialización), su gradiente será siempre cero y sus pesos nunca se actualizarán, volviéndose "muerta".
    *   **Leaky ReLU**: Una solución que permite que las neuronas fuera del intervalo activo aún propaguen algo de gradiente hacia atrás mediante un pequeño parámetro `α` (`Φ(v) = αv` si `v ≤ 0`, `v` en otro caso).
    *   **Maxout**: Una solución más reciente que usa dos vectores de coeficientes `W1` y `W2`, y la activación es `max{W1·X, W2·X}`. Es una generalización de ReLU, no satura y es lineal casi en todas partes, aunque duplica el número de parámetros.

### Estrategias de Descenso de Gradiente
El descenso de gradiente más pronunciado puede ser ineficiente debido a oscilaciones y "zigzagueos", especialmente en superficies de pérdida con alta curvatura.

*   **Decaimiento de la Tasa de Aprendizaje (Learning Rate Decay)**: Una tasa de aprendizaje constante es problemática (lento al inicio o oscilación/divergencia al final). El decaimiento de la tasa de aprendizaje a lo largo del tiempo ajusta esto naturalmente. Las funciones comunes son el decaimiento exponencial (`αt = α0 * exp(-k * t)`) y el decaimiento inverso (`αt = α0 / (1 + k * t)`). También se puede reducir la tasa cada cierto número de épocas (step decay) o cuando la pérdida en validación deja de mejorar.
*   **Aprendizaje Basado en Momento (Momentum-Based Learning)**: Reconoce que el zigzagueo es resultado de pasos contradictorios. La idea es moverse en una dirección "promediada" de los últimos pasos para suavizar el zigzagueo. Un parámetro de suavizado `β` (0,1) ayuda a la velocidad `V` a ser consistente. Acelera el aprendizaje al preferir direcciones consistentes, permitiendo pasos más grandes sin "explosiones" laterales. Ayuda a sortear regiones planas de la superficie de pérdida y a evitar mínimos locales al "sobrepasar" ligeramente el objetivo.
*   **Momento de Nesterov (Nesterov Momentum)**: Una modificación que calcula los gradientes en un punto que se alcanzaría si se ejecutara una versión `β`-descontada del paso anterior (`W + βV`). Esto incorpora información sobre cómo cambiarán los gradientes debido al momento, lo que puede conducir a una convergencia más rápida. Funciona mejor con mini-lotes de tamaño modesto.
*   **Tasas de Aprendizaje Específicas de Parámetro**: Tienen diferentes tasas de aprendizaje para diferentes parámetros para acelerar las actualizaciones, especialmente donde los gradientes son inconsistentes o pequeños.
    *   **AdaGrad**: Mantiene un seguimiento de la magnitud cuadrada agregada del gradiente para cada parámetro. Divide el gradiente por la raíz cuadrada de esta suma acumulativa (`wi ← wi - α/√Ai * ∂L/∂wi`). Esto penaliza movimientos a lo largo de direcciones que fluctúan salvajemente y enfatiza movimientos consistentes. Sin embargo, la suma acumulativa hace que la tasa de aprendizaje disminuya continuamente, ralentizando el progreso prematuramente.
    *   **RMSProp**: Similar a AdaGrad, pero utiliza un **promedio exponencial** de los gradientes cuadrados en lugar de la suma acumulativa. Esto evita la ralentización prematura y el problema de gradientes "antiguos" de AdaGrad, ya que la importancia de los gradientes pasados decae exponencialmente.
    *   **AdaDelta **: Una variante de RMSProp que **elimina la necesidad de un parámetro de tasa de aprendizaje global** al calcularlo en función de las actualizaciones incrementales previas.
    *   **Adam **: Combina la normalización "señal-ruido" de AdaGrad/RMSProp con el suavizado exponencial del gradiente de primer orden para incorporar momento. También aborda el **sesgo inherente en el suavizado exponencial** durante las primeras iteraciones. Es **extremadamente popular** y a menudo rinde competitivamente.
*   **Acantilados (Cliffs) e Inestabilidad de Orden Superior**: Las superficies de pérdida pueden tener regiones con pendientes suaves que cambian abruptamente a "acantilados". Los gradientes de primer orden pueden no ser suficientes para controlar el tamaño de la actualización, llevando a sobrepasos o subidas excesivas. Los parámetros compartidos en redes recurrentes pueden causar estos efectos de orden superior.
*   **Recorte de Gradiente (Gradient Clipping)**: Una técnica para manejar gradientes con magnitudes excesivamente diferentes. Puede ser:
    *   **Recorte basado en valor**: Limita los valores de los gradientes individuales a un umbral mínimo y máximo.
    *   **Recorte basado en la norma**: Normaliza el vector de gradientes completo por su norma L2.
    Es particularmente efectivo para evitar el **problema de gradiente explotado en redes neuronales recurrentes**.
*   **Derivadas de Segundo Orden**:
    *   **Matriz Hessiana (H)**: Contiene las segundas derivadas parciales de la función de pérdida con respecto a todos los pares de parámetros. Los métodos de segundo orden aproximan la superficie de pérdida local con un cuenco cuadrático, lo que es más preciso que una aproximación lineal.
    *   **Método de Newton**: Utiliza la inversa de la Hessiana (`W* ← W0 - H^-1[∇L(W0)]`). A diferencia de los métodos de primer orden, no necesita una tasa de aprendizaje explícita y puede converger en un solo paso para funciones cuadráticas. Sesga los pasos de aprendizaje hacia direcciones de baja curvatura, donde el gradiente cambia menos. Es ventajoso en valles estrechos, donde el descenso de gradiente de primer orden rebota violentamente.
    *   **Problemas del Método de Newton**: La Hessiana es demasiado grande para almacenar o calcular explícitamente en redes neuronales grandes.
    *   **Gradientes Conjugados y Optimización sin Hessiana (Hessian-Free Optimization)**: Requiere `d` pasos para alcanzar el óptimo de una función cuadrática (en lugar de un solo paso de Newton). Genera direcciones de movimiento donde el trabajo hecho en iteraciones anteriores no se deshace. El cálculo de la dirección de búsqueda puede hacerse **sin la computación explícita de la Hessiana**, utilizando diferencias finitas.
    *   **Métodos Cuasi-Newton y BFGS**: Aproximan la inversa de la Hessiana (`Gt`) iterativamente. **BFGS** (Broyden-Fletcher-Goldfarb-Shanno) actualiza `Gt` con actualizaciones de bajo rango. **L-BFGS** (Limited-memory BFGS) reduce drásticamente el requisito de memoria al no llevar la matriz `Gt` de la iteración anterior, almacenando solo los `m` vectores más recientes de cambio de parámetros y cambio de gradiente.
    *   **Problemas con Puntos de Silla**: Los métodos de segundo orden son susceptibles a los **puntos de silla** (puntos estacionarios con gradiente cero que no son mínimos ni máximos). A diferencia de los mínimos locales, los puntos de silla son frecuentes en redes neuronales profundas y pueden atrapar los métodos de segundo orden. Los métodos de primer orden a menudo logran escapar de ellos.
*   **Promediado de Polyak (Polyak Averaging)**: Una técnica para estabilizar cualquier algoritmo de aprendizaje, promediando exponencialmente los parámetros a lo largo del tiempo para evitar el comportamiento de rebote.
*   **Mínimos Locales y Espurios**: La función objetivo de una red neuronal no es convexa y es probable que tenga muchos mínimos locales. Sin embargo, los mínimos locales de redes reales a menudo tienen valores de función objetivo muy similares al mínimo global, lo que reduce su impacto negativo. Los **mínimos espurios** son un problema mayor, causados por la insuficiencia de datos de entrenamiento que no se generalizan bien a los datos de prueba no vistos.

### Normalización por Lotes (Batch Normalization)
La normalización por lotes es un método reciente para abordar los problemas de **gradiente desvanecido y explotado**, y el problema de **cambio de covarianza interno** (internal covariate shift).

*   **Mecanismo**: Añade **"capas de normalización" adicionales entre las capas ocultas** para que las características tengan una varianza similar. Cada unidad de normalización contiene dos parámetros adicionales `βi` (media) y `γi` (desviación estándar) que se aprenden de forma dirigida por los datos. `βi` actúa como una variable de sesgo aprendida.
*   **Ubicación**: Se discuten dos opciones: después de la función de activación (post-activación) o **antes de la función de activación (pre-activación)**. Esta última se argumenta como más ventajosa y es el foco de la exposición.
*   **Transformaciones**: Para un mini-lote de `m` instancias, se calcula la media `µi` y la desviación estándar `σi` para la unidad `i`. Luego, las activaciones se normalizan (`v̂(r)i = (v(r)i - µi) / σi`) y se escalan/desplazan con `γi` y `βi` (`a(r)i = γi * v̂(r)i + βi`).
*   **Backpropagation a través de BN**: El algoritmo de retropropagación se ajusta para contabilizar este nodo especial, incluyendo la optimización de los parámetros `βi` y `γi`.
*   **Inferencia (Testing)**: Durante la predicción, los valores `µi` y `σi` se calculan previamente utilizando todo el conjunto de datos de entrenamiento y se tratan como constantes.
*   **Efecto Regularizador**: La normalización por lotes también actúa como un regularizador, ya que el mismo punto de datos puede causar actualizaciones ligeramente diferentes dependiendo del lote en el que se incluya, lo que añade una especie de "ruido" al proceso. Experimentalmente, a menudo hace que otros métodos de regularización como Dropout sean menos efectivos.

### Trucos Prácticos para la Aceleración y Compresión
El entrenamiento de redes neuronales es computacionalmente costoso, y el despliegue requiere eficiencia y poca memoria.

*   **Aceleración con GPU**: Las **Unidades de Procesamiento Gráfico (GPUs)** son altamente eficientes para operaciones repetitivas de matrices (como las multiplicaciones matriciales extensivamente usadas en redes neuronales), gracias a su alto ancho de banda de memoria y su arquitectura multinúcleo con multithreading (SIMT). Bibliotecas como NVIDIA cuDNN  abstraen la programación de bajo nivel, permitiendo que el mismo código se ejecute en CPU o GPU con mínimas modificaciones.
*   **Implementaciones Paralelas y Distribuidas**:
    *   **Paralelismo de Hiperparámetros**: Entrenar redes con diferentes configuraciones de parámetros en diferentes procesadores de forma independiente, sin comunicación, para encontrar la mejor combinación.
    *   **Paralelismo de Modelo**: Dividir un modelo demasiado grande para una sola GPU entre varias GPUs. Cada GPU trabaja en la misma parte del lote de entrenamiento, requiriendo comunicación entre GPUs para intercambiar activaciones y gradientes.
    *   **Paralelismo de Datos**: El modelo cabe en cada GPU, pero los datos son muy grandes. Los parámetros se comparten entre GPUs, cada una procesando diferentes puntos de entrenamiento. El **descenso de gradiente estocástico asíncrono** con un servidor de parámetros es popular en entornos industriales a gran escala, ya que evita los cuellos de botella de los mecanismos de bloqueo.
    *   **Paralelismo Híbrido**: Combina paralelismo de datos (para capas tempranas con más cómputo) y paralelismo de modelo (para capas posteriores con más parámetros, como las capas totalmente conectadas).
*   **Trucos Algorítmicos para la Compresión de Modelos (en despliegue)**:
    *   **Esparsificación de Pesos (Poda)**: Eliminar los pesos con valores absolutos pequeños, seguido de un reajuste de los pesos restantes. La regularización L1 puede fomentar pesos cero. Esto puede reducir significativamente el tamaño del modelo sin pérdida de precisión, mejorando el rendimiento de la caché.
    *   **Aprovechamiento de Redundancias (Aproximación de Rango Bajo)**: Aproximar matrices de pesos `W` como `U V^T` (donde `U` y `V` son matrices mucho más pequeñas) después del entrenamiento. Esto reduce el número de parámetros al aprovechar la redundancia en los pesos y características, que a menudo co-adaptan durante el entrenamiento.
    *   **Compresión Basada en Hash**: Forzar que entradas de la matriz de pesos elegidas aleatoriamente compartan los mismos valores de parámetros mediante una función hash. Reduce drásticamente el espacio requerido manteniendo la expresividad del modelo.
    *   **Modelos Mimic (Destilación de Conocimiento)**: Entrenar un modelo más pequeño y superficial (el "modelo estudiante" o "mimic") utilizando un nuevo conjunto de datos de entrenamiento creado a partir de un modelo más grande ya entrenado (el "modelo maestro"). El modelo maestro genera "etiquetas suaves" (probabilidades softmax) para ejemplos sin etiquetar. Este enfoque puede lograr una precisión similar con un modelo mucho más pequeño, ya que el maestro simplifica las regiones complejas, elimina errores de etiquetado y proporciona objetivos más informativos y dependientes de las entradas disponibles, actuando como una forma de regularización.

En resumen, el entrenamiento de redes neuronales profundas implica un balance entre la complejidad del modelo, la eficiencia computacional y la estabilidad de los gradientes. La retropropagación es el pilar, pero su eficacia depende de una cuidadosa inicialización, pre-procesamiento de datos y la elección de estrategias de descenso de gradiente que mitiguen problemas como los gradientes desvanecidos/explotados y los puntos de silla. La normalización por lotes y diversas técnicas de aceleración/compresión son cruciales para la aplicación práctica de estas redes.

***

# Parte IV

***

# Generalización y Regularización en Redes Profundas

### Introducción a la Generalización en Redes Neuronales Profundas

Las redes neuronales son aprendices potentes, capaces de aprender funciones complejas en diversas áreas. Sin embargo, esta gran capacidad también es su mayor debilidad, ya que a menudo se **sobreajustan (overfit)** a los datos de entrenamiento si el proceso de aprendizaje no se diseña con cuidado. El sobreajuste implica que una red neuronal ofrece un rendimiento de predicción excelente en los datos de entrenamiento, pero se desempeña mal en instancias de prueba no vistas. Esto sucede porque el proceso de aprendizaje tiende a recordar **artefactos aleatorios** de los datos de entrenamiento que no se generalizan bien a los datos de prueba. En sus formas extremas, el sobreajuste se conoce como memorización.

La **generalización** es la capacidad de un aprendiz para proporcionar predicciones útiles para instancias que no ha visto antes. Es el "santo grial" en todas las aplicaciones de aprendizaje automático, ya que el objetivo es predecir etiquetas para ejemplos nuevos, no los que ya están etiquetados.

El nivel de sobreajuste depende tanto de la **complejidad del modelo** como de la **cantidad de datos disponibles**. Los modelos más complejos, con un mayor número de parámetros, tienen más grados de libertad y pueden explicar puntos específicos de los datos de entrenamiento sin generalizar bien a puntos no vistos. Por ejemplo, un modelo polinomial de grado alto puede ajustarse exactamente a un pequeño conjunto de datos de entrenamiento con error cero, pero esto no garantiza un error cero en datos de prueba no vistos.

### La Compensación Sesgo-Varianza (Bias-Variance Trade-Off)

La comprensión de la generalización en redes neuronales se basa en la **compensación sesgo-varianza**. Esta compensación establece que el error cuadrático de un algoritmo de aprendizaje puede dividirse en tres componentes:

*   **Sesgo (Bias):** Es el error causado por las suposiciones simplificadoras en el modelo, lo que lleva a **errores consistentes** en ciertas instancias de prueba a través de diferentes conjuntos de datos de entrenamiento. Un modelo con alto sesgo no puede ajustarse con precisión a la distribución de datos subyacente, incluso con una cantidad infinita de datos. Por ejemplo, un modelo lineal tiene un sesgo mayor que un modelo polinomial para datos curvos, ya que nunca podrá ajustarse exactamente a la curva, sin importar cuántos datos tenga. El sesgo no se puede eliminar completamente.
*   **Varianza (Variance):** Es causada por la incapacidad de aprender todos los parámetros del modelo de una manera estadísticamente robusta, especialmente cuando los datos son limitados y el modelo tiene un gran número de parámetros. La alta varianza se manifiesta como **sobreajuste a los datos de entrenamiento específicos**, lo que provoca predicciones muy diferentes para la misma instancia de prueba si se usan diferentes conjuntos de datos de entrenamiento. Los modelos con alta varianza tienden a memorizar artefactos aleatorios de los datos de entrenamiento.
*   **Ruido (Noise):** Es el error inherente en los datos mismos, que no puede ser eliminado por el modelo.

La varianza es el término clave que impide que las redes neuronales generalicen bien. En general, las redes neuronales con un gran número de parámetros tendrán una mayor varianza. Por otro lado, muy pocos parámetros pueden causar sesgo. Existe un punto de **complejidad óptima del modelo** donde el rendimiento se optimiza. La escasez de datos de entrenamiento aumentará la varianza.

### Detección del Sobreajuste

Existen varias señales claras de sobreajuste:
1.  **Predicciones inconsistentes:** Un mismo ejemplo de prueba produce predicciones muy diferentes cuando el modelo se entrena con distintos conjuntos de datos. Esto indica que el proceso de entrenamiento está memorizando las particularidades del conjunto de datos específico en lugar de aprender patrones generalizables.
2.  **Gran diferencia entre el error de entrenamiento y el error de prueba:** El rendimiento del modelo es excelente en los datos de entrenamiento, pero significativamente peor en los datos no vistos. Por ejemplo, una precisión de entrenamiento cercana al 100% en un conjunto de entrenamiento pequeño, mientras que el error de prueba es bastante bajo, es una señal de sobreajuste.

Si se detecta sobreajuste (un gran desfase entre la precisión de entrenamiento y la de prueba), la primera solución es **recopilar más datos**. Con más datos de entrenamiento, la precisión de entrenamiento se reducirá y la precisión de prueba/validación aumentará. Sin embargo, si no hay más datos disponibles, se deben utilizar otras técnicas para mejorar la generalización.

### Problemas de Generalización en la Optimización y Evaluación del Modelo

Para evitar la sobreestimación de la precisión y garantizar una buena generalización, los datos etiquetados deben dividirse en tres partes:

*   **Datos de entrenamiento:** Utilizados para construir el modelo (aprender los pesos de la red neuronal).
*   **Datos de validación:** Utilizados para la selección del modelo y el ajuste de hiperparámetros (ej., tasa de aprendizaje, número de capas, función de activación). Estos datos actúan como un conjunto de prueba para afinar los parámetros.
*   **Datos de prueba:** Utilizados para evaluar la precisión final del modelo ya ajustado. Es crucial que estos datos no se utilicen en absoluto durante el entrenamiento o el ajuste de parámetros para evitar la contaminación y obtener una evaluación optimista. Se usan solo una vez, al final del proceso.

La división convencional de los datos suele ser 2:1:1 para entrenamiento, validación y prueba, aunque en la era moderna con grandes conjuntos de datos, divisiones como 98:1:1 son comunes, usando la mayor parte para el entrenamiento y una cantidad modesta (constante) para validación y prueba.

Métodos de evaluación:
*   **Hold-Out:** Una fracción de instancias se usa para entrenar, y el resto (held-out) para probar. Es simple y eficiente, popular en configuraciones a gran escala. Sin embargo, puede subestimar la verdadera precisión y causar un sesgo pesimista si la distribución de clases en el conjunto de prueba es diferente a la del entrenamiento.
*   **Validación Cruzada (Cross-Validation):** El conjunto de datos se divide en *q* segmentos iguales. Uno se usa para prueba y los (*q*-1) restantes para entrenamiento, repitiendo el proceso *q* veces. La precisión promedio de las *q* pruebas se reporta. Este método puede estimar la precisión verdadera con más exactitud que Hold-Out cuando *q* es grande. Sin embargo, es costoso computacionalmente y, por lo tanto, rara vez se usa en redes neuronales debido a problemas de eficiencia, especialmente con grandes conjuntos de datos.

### Métodos Clave para Evitar el Sobreajuste (Mejorar la Generalización)

Aunque una forma natural de evitar el sobreajuste es construir redes más pequeñas, a menudo se ha observado que es **mejor construir redes grandes y luego regularizarlas**. Las redes grandes mantienen la opción de construir un modelo más complejo si es necesario, mientras que la regularización puede suavizar los artefactos aleatorios.

Aquí se detallan los principales métodos:

1.  **Regularización Basada en Penalidades (Penalty-based Regularization)**
    Es la técnica más común. Implica crear una penalidad o restricciones sobre los parámetros para favorecer modelos más simples. En lugar de reducir los parámetros de forma "dura", se usa una penalidad "suave".

    *   **Regularización L2 (Tikhonov Regularization):** Se añade la suma de los cuadrados de los valores de los parámetros (λ Σwᵢ²) a la función de pérdida. Esto es aproximadamente equivalente a **multiplicar cada parámetro wᵢ por un factor de decaimiento multiplicativo de (1 − αλ)** antes de cada actualización, donde α es la tasa de aprendizaje. Actúa como un **mecanismo de olvido** que acerca los pesos a sus valores iniciales, impidiendo que el modelo memorice los datos de entrenamiento y asegurando que solo las actualizaciones repetidas tengan un efecto significativo.
    *   **Regularización L1:** En lugar de los cuadrados, se penaliza la suma de los valores absolutos de los coeficientes (λ Σ|wᵢ|). A diferencia de L2, L1 usa **actualizaciones aditivas** como mecanismo de olvido. Una propiedad interesante de L1 es que crea **soluciones dispersas**, donde la mayoría de los valores de *wᵢ* son cero. Esto significa que actúa como un **selector de características**, ya que los inputs con peso cero no afectan la predicción. También puede resultar en redes neuronales dispersas, donde las conexiones con pesos cero pueden eliminarse, lo que las hace más eficientes para reentrenar. Sin embargo, en términos de precisión, L2 generalmente supera a L1.
    *   **Penalización de Unidades Ocultas (Sparse Representations):** En lugar de penalizar los parámetros, se penalizan las activaciones de las unidades ocultas para que solo un pequeño subconjunto de neuronas se active para cualquier instancia de datos. Esto se logra comúnmente aplicando una **penalidad L1 a las unidades ocultas** (λ Σ|hᵢ|). Esto lleva a representaciones dispersas y puede modificar el algoritmo de retropropagación para incorporar la penalidad.

    **Conexiones con la Inyección de Ruido:** La adición de ruido a la entrada tiene conexiones con la regularización basada en penalidades. Se ha demostrado que la adición de una cantidad igual de ruido gaussiano a cada entrada es **equivalente a la regularización de Tikhonov** para una red neuronal de una sola capa con función de activación de identidad (regresión lineal). Incluso en redes neuronales con activaciones no lineales, la penalización sigue siendo intuitivamente similar a la adición de ruido. La inyección de ruido es una forma de regularización.

2.  **Métodos de Conjunto (Ensemble Methods)**
    Estos métodos se inspiran en la compensación sesgo-varianza y se enfocan principalmente en la **reducción de la varianza**. Las redes neuronales, al ser capaces de construir modelos complejos, tienden a tener baja varianza, pero alta varianza, lo que se manifiesta como sobreajuste.

    *   **Bagging y Submuestreo (Subsampling):**
        *   **Bagging:** Se crean múltiples conjuntos de datos de entrenamiento **muestreando con reemplazo** del conjunto original (el tamaño de la muestra *s* puede ser igual al tamaño del conjunto de entrenamiento *n*, o menor). Se entrena un modelo en cada conjunto remuestreado, y las predicciones de estos *m* modelos se promedian para obtener una predicción final más robusta.
        *   **Submuestreo:** Similar al Bagging, pero los conjuntos de entrenamiento se crean **sin reemplazo**. Es esencial elegir *s < n* para obtener diferentes resultados. El submuestreo es preferible cuando se dispone de suficientes datos de entrenamiento, mientras que Bagging es útil cuando los datos son limitados.
        *   Ninguno puede eliminar toda la varianza debido a la correlación positiva entre las predicciones de los modelos.

    *   **Selección y Promediado de Modelos Paramétricos:** Consiste en entrenar múltiples configuraciones de redes neuronales (variando hiperparámetros, capas, funciones de activación) y luego seleccionar la que tenga mejor rendimiento en el conjunto de validación (selección de modelos). Una mejora para reducir la varianza es **seleccionar las *k* mejores configuraciones y promediar sus predicciones**.
    *   **Eliminación Aleatoria de Conexiones (Randomized Connection Dropping):** Se eliminan conexiones aleatoriamente entre capas para crear modelos diversos. Las predicciones promediadas de estos modelos suelen ser muy precisas. A diferencia de Dropout, los pesos de los diferentes modelos no se comparten.
    *   **Dropout:** Un método que utiliza el **muestreo de nodos** (en lugar de bordes/conexiones) para crear un conjunto de redes neuronales. Si un nodo se elimina, todas sus conexiones entrantes y salientes también se eliminan. Los nodos se muestrean de las capas de entrada y ocultas con una cierta probabilidad (ej., entre 20% y 50%).
        *   La clave es que los **pesos de las diferentes redes muestreadas se comparten**. El entrenamiento implica muestrear una sub-red para cada mini-lote de instancias de entrenamiento y actualizar los pesos de los bordes retenidos mediante retropropagación.
        *   **Regla de inferencia de escalado de pesos:** Para la predicción en tiempo de prueba, no es necesario evaluar todas las subredes. En su lugar, se realiza la propagación hacia adelante en la red base completa (sin eliminación), **re-escalando los pesos** de cada unidad al multiplicarlos por la probabilidad de muestreo de esa unidad. Esta regla es una heurística que funciona bien en la práctica, aunque no es exacta para redes no lineales.
        *   **Efecto:** Dropout actúa como un **regularizador** al incorporar ruido de enmascaramiento (estableciendo algunas unidades a 0) en las entradas y representaciones ocultas. Esto previene la **co-adaptación de características** entre unidades ocultas, forzando una redundancia entre las características aprendidas y promoviendo que subconjuntos más pequeños de características tengan poder predictivo. Esto conduce a una mayor robustez y generalización.
        *   Dropout es eficiente porque cada subred muestreada se entrena con un pequeño conjunto de ejemplos. Requiere usar modelos más grandes y más unidades para obtener todos sus beneficios, lo que genera una sobrecarga computacional oculta.
    *   **Conjuntos de Perturbación de Datos (Data Perturbation Ensembles):** Se añade explícitamente una pequeña cantidad de ruido a los datos de entrada, y los pesos se aprenden en estos datos perturbados. Este proceso se repite varias veces, y las predicciones se promedian. Es un método genérico que también se usa en autoencoders de eliminación de ruido (de-noising autoencoders). También se pueden agregar ruido a las capas ocultas o realizar aumentos de datos (rotaciones, traducciones en imágenes).

3.  **Detención Temprana (Early Stopping)**
    Consiste en **terminar el método de optimización iterativo antes de que converja completamente** en los datos de entrenamiento. El punto de parada se determina monitoreando el error del modelo en un **conjunto de validación separado**. Se detiene el entrenamiento cuando el error en el conjunto de validación comienza a aumentar, incluso si el error en el conjunto de entrenamiento sigue disminuyendo. Esto se debe a que las últimas etapas del entrenamiento a menudo sobreajustan las sutilezas específicas de los datos de entrenamiento, que no generalizan bien a los datos de prueba.

    La detención temprana actúa como una **restricción en el proceso de optimización**, limitando la distancia de la solución final desde el punto de inicialización, lo que es una forma de regularización. Es fácil de implementar y se puede combinar con otros regularizadores.

4.  **Preentrenamiento (Pretraining)**
    Es una forma de aprendizaje en la que un algoritmo "codicioso" (greedy) se utiliza para encontrar una **buena inicialización** de los pesos. Los pesos en diferentes capas de la red neuronal se entrenan secuencialmente. Estos pesos entrenados sirven como un buen punto de partida para el proceso global de aprendizaje y se ajustan finamente con retropropagación tradicional.

    *   **Preentrenamiento no supervisado:** Fue un avance fundamental. Se entrena la red capa por capa de forma codiciosa, típicamente usando **autoencoders**. Los autoencoders aprenden a reconstruir sus entradas, lo que les permite capturar patrones y características repetidas en los datos de manera no supervisada. Estos pesos inicializados de forma no supervisada ayudan a las redes profundas a superar problemas como los gradientes que explotan o se desvanecen, que pueden causar que las capas iniciales no se entrenen adecuadamente. Actúa como una **forma indirecta de regularización** y mejora la generalización al precargar el proceso de entrenamiento en una región semánticamente relevante del espacio de parámetros. A menudo, las representaciones aprendidas están suavemente relacionadas con las etiquetas de clase, lo que permite un ajuste fino posterior más efectivo.
    *   **Preentrenamiento supervisado:** Aunque es posible, a menudo no ofrece resultados tan buenos como el preentrenamiento no supervisado en algunos entornos, especialmente en términos de error de generalización en datos de prueba no vistos. Esto se debe a que el preentrenamiento supervisado puede ser "demasiado codicioso", inicializando las capas tempranas de manera demasiado directa a los resultados, lo que no explota las ventajas de la profundidad tan bien como el preentrenamiento no supervisado.

5.  **Métodos de Continuación y Currículum (Continuation and Curriculum Methods)**
    Estos métodos se basan en la idea de que es más fácil entrenar modelos simples sin sobreajuste, y que comenzar con el punto óptimo de un modelo simple proporciona una buena inicialización para un modelo complejo relacionado. Trabajan del **simple al complejo**.

    *   **Aprendizaje por Continuación (Continuation Learning):** Se inicia con una versión simplificada del problema de optimización, se resuelve, y luego se refina gradualmente hacia el problema complejo. Se enfoca en una **vista centrada en el modelo** (model-centric), diseñando una serie de funciones de pérdida que aumentan en dificultad. La "suavización" de la función de pérdida es una forma de regularización.
    *   **Aprendizaje por Currículum (Curriculum Learning):** Se comienza entrenando el modelo con **instancias de datos más simples**, añadiendo gradualmente instancias más difíciles al conjunto de entrenamiento. Se basa en una **vista centrada en los datos** (data-centric). Los ejemplos fáciles "preentrenan" al aprendiz hacia una configuración de parámetros razonable, y luego se incluyen ejemplos difíciles (a menudo ruidosos o excepcionales) en las iteraciones posteriores, pero siempre manteniendo una mezcla para evitar el sobreajuste a solo los ejemplos difíciles.

6.  **Compartir Parámetros con Conocimientos de Dominio (Sharing Parameters with Domain-Specific Insights)**
    Reduce la cantidad de grados de libertad del modelo y se habilita a menudo por conocimientos específicos del dominio.
    *   **Autoencoders:** Los pesos simétricos en las porciones de codificador y decodificador a menudo se comparten, lo que mejora las propiedades de regularización.
    *   **Redes Neuronales Recurrentes (RNNs):** Utilizadas para datos secuenciales (ej., texto, series de tiempo). Los parámetros se comparten entre diferentes capas que representan diferentes marcas de tiempo, asumiendo que el mismo modelo se usa en cada paso de tiempo.
    *   **Redes Neuronales Convolucionales (CNNs):** Utilizadas para reconocimiento de imágenes. Los pesos se comparten en parches contiguos de la red, basándose en la idea de que una característica (ej., un borde) debería interpretarse de la misma manera sin importar su ubicación en la imagen. Estas técnicas usan conocimientos semánticos para reducir la huella de parámetros y dispersar las conexiones.
    *   **Compartir pesos suave (Soft Weight Sharing):** Los parámetros no están completamente atados, pero se asocia una penalización si son diferentes, empujándolos a ser similares (ej., λ(wᵢ - wⱼ)²/2).

### Regularización en Aplicaciones No Supervisadas

Aunque el sobreajuste es menos problemático en aplicaciones no supervisadas (ya que un ejemplo de entrenamiento contiene más bits de información en sus múltiples dimensiones), la regularización sigue siendo beneficiosa para **imponer una estructura deseada en las representaciones aprendidas**.

1.  **Penalización Basada en Valores: Autoencoders Escasos (Sparse Autoencoders)**
    Los autoencoders escasos tienen un número de unidades ocultas mayor que las unidades de entrada (overcomplete), pero los valores de las unidades ocultas se **incentivan a ser cero** mediante penalización explícita (ej., penalidad L1) o restricciones. Esto resulta en que la mayoría de los valores en las unidades ocultas sean cero en la convergencia, creando **representaciones dispersas**.

2.  **Inyección de Ruido: Autoencoders de Eliminación de Ruido (De-noising Autoencoders)**
    Se basa en la inyección de ruido en lugar de la penalización de pesos o unidades ocultas. El objetivo es **reconstruir ejemplos limpios a partir de datos de entrenamiento corrompidos**. Se pueden añadir diferentes tipos de ruido:
    *   **Ruido Gaussiano:** Para entradas de valores reales.
    *   **Ruido de Enmascaramiento:** Se establece una fracción de entradas a cero (útil para entradas binarias).
    *   **Ruido Salt-and-pepper:** Una fracción de entradas se establece a sus valores mínimos o máximos posibles.
    El autoencoder aprende el "verdadero manifold" (variedad) en el que los datos están incrustados, proyectando cada punto corrompido a su punto más cercano en este manifold. La adición de ruido actúa como un excelente regularizador, mejorando el rendimiento en entradas fuera de muestra.

3.  **Penalización Basada en Gradientes: Autoencoders Contractivos (Contractive Autoencoders)**
    Son autoencoders fuertemente regularizados donde no se desea que la representación oculta cambie significativamente con pequeños cambios en los valores de entrada. Penalizan las **derivadas parciales de los valores ocultos con respecto a las entradas**. Esto significa que son insensibles a los cambios en la entrada que son inconsistentes con la estructura del manifold de los datos, "amortiguando" el ruido. La penalización se añade a la función de pérdida junto con el error de reconstrucción. La diferencia con los autoencoders de eliminación de ruido es que los autoencoders contractivos logran la robustez analíticamente mediante un término de regularización, mientras que los de eliminación de ruido lo hacen estocásticamente añadiendo ruido explícitamente. Los autoencoders contractivos son más útiles para la ingeniería de características, ya que la responsabilidad de la regularización recae únicamente en el codificador.

4.  **Estructura Probabilística Oculta: Autoencoders Variacionales (Variational Autoencoders - VAEs)**
    Imponen una **estructura probabilística específica** en las unidades ocultas, generalmente que las activaciones en las unidades ocultas (sobre el conjunto de datos completo) se extraigan de una **distribución gaussiana estándar** (media cero, varianza unitaria).
    *   **Mecanismo:** El codificador no produce directamente la representación oculta, sino que produce los vectores de media (µ(X)) y desviación estándar (σ(X)) de una distribución Gaussiana condicional para una entrada *X*. La representación oculta real *h(X)* se **muestrea** a partir de esta distribución, combinando el muestreo de una distribución gaussiana estándar con la escala y traslación por µ(X) y σ(X). Esta "reparametrización" hace que el modelo sea diferenciable y entrenable mediante retropropagación.
    *   **Función de Pérdida:** La función de pérdida total *J* es una suma ponderada del **error de reconstrucción** (ej., error cuadrático entre la entrada y su reconstrucción) y la **pérdida de regularización**. La pérdida de regularización es la **divergencia de Kullback-Leibler (KL)** entre la distribución oculta condicional (µ(X), σ(X)) y la distribución gaussiana estándar. Esta penalidad fuerza a las representaciones ocultas a ser estocásticas, alentando que cada entrada se mapee a su propia región estocástica en el espacio oculto en lugar de a un solo punto, lo que aumenta el poder de generalización.
    *   **Generación de Muestras:** Una aplicación interesante es la **generación de nuevas muestras realistas**. Después del entrenamiento, se puede descartar el codificador y simplemente alimentar muestras de la distribución Gaussiana estándar al decodificador para generar nuevas instancias de datos. Esto funciona porque la regularización de VAEs fomenta que los puntos de entrenamiento se distribuyan aproximadamente de forma Gaussiana en el espacio latente, reduciendo las discontinuidades y permitiendo transiciones suaves entre clases al "caminar" por este espacio.
    *   **Autoencoders Variacionales Condicionales (Conditional VAEs):** Permiten añadir una entrada condicional (un contexto, ej., una imagen dañada) para guiar la generación o reconstrucción de los datos faltantes.
    *   **Relación con Redes Generativas Antagónicas (GANs):** Los VAEs están relacionados con los GANs en su capacidad de generar imágenes similares a un conjunto de datos base y para completar datos faltantes. Sin embargo, los GANs a menudo producen resultados más realistas y menos borrosos, ya que entrenan explícitamente un "generador" para crear "falsificaciones" que engañen a un "discriminador", lo que fomenta una mayor creatividad.

En resumen, la generalización es fundamental en las redes neuronales profundas y se logra combatiendo el sobreajuste. Esto se hace a través de diversas técnicas de regularización que, en lugar de restringir rígidamente la complejidad del modelo, permiten que redes más grandes y potentes se ajusten de manera flexible a los datos, suavizando los artefactos aleatorios y mejorando el rendimiento en instancias no vistas.

***

# Parte V

***

# Redes de Función de Base Radial: Teoría y Aplicación

Las Redes de Funciones de Base Radial (RBF) representan una arquitectura de red neuronal fundamentalmente distinta de las redes de alimentación directa (feed-forward) tradicionales. A diferencia de estas últimas, que transmiten las entradas capa por capa para crear salidas finales y logran la no linealidad mediante la composición repetida de funciones de activación, una red RBF generalmente utiliza solo una capa de entrada, una única capa oculta (con un comportamiento especial definido por funciones RBF) y una capa de salida.

Aquí un análisis detallado de las Redes de Funciones de Base Radial:

*   **Arquitectura y Computaciones**
    *   **Capa de Entrada**: Simplemente transmite las características de entrada a las capas ocultas. No realiza cálculos y el número de unidades de entrada es igual a la dimensionalidad de los datos.
    *   **Capa Oculta (RBF Activation)**: Es el corazón de la red RBF.
        *   Realiza una computación basada en la comparación con **vectores prototipo** (µi). Cada unidad oculta contiene un vector prototipo y un ancho de banda (σi).
        *   Transforma los puntos de entrada, donde la estructura de clase podría no ser linealmente separable, en un nuevo espacio que a menudo sí lo es. Esta transformación es no lineal y a menudo aumenta la dimensionalidad, basándose en el teorema de Cover sobre la separabilidad de patrones.
        *   La activación (Φi(X)) de la i-ésima unidad oculta se define por una función de base radial, comúnmente la función Gaussiana: `hi = Φi(X) = exp(−||X − µi||² / (2 · σi²))`.
        *   Cada unidad oculta busca tener una alta influencia en el clúster de puntos más cercanos a su vector prototipo. El número de unidades ocultas (`m`) puede verse como el número de clústeres utilizados para el modelado.
    *   **Capa de Salida**: Utiliza modelado de clasificación o regresión lineal con respecto a las entradas de la capa oculta. Las conexiones de la capa oculta a la capa de salida tienen pesos (`wi`) asociados. La predicción (ŷ) se define como la suma ponderada de las activaciones de la capa oculta: `ŷ = Σ wihi`. Los cálculos son similares a los de una red de alimentación directa estándar.
    *   **Neurona de polarización (Bias Neuron)**: La capa oculta puede contener neuronas de polarización, que pueden implementarse como una unidad oculta siempre activa o con un ancho de banda infinito (σi = ∞).

*   **Principios Clave y Funcionalidad**
    *   **Transformación para la Separabilidad Lineal**: El objetivo principal de la capa oculta es crear una transformación que promueva la separabilidad lineal, lo que permite que incluso los clasificadores lineales funcionen bien en los datos transformados. Esto se ilustra con ejemplos donde el perceptrón tradicional falla en el espacio de entrada, pero la transformación RBF lo resuelve en un espacio oculto de mayor dimensión.
    *   **Características Locales**: Las RBFs Gaussianas con un ancho de banda pequeño activan solo un número limitado de unidades ocultas en regiones locales específicas, lo que conduce a representaciones dispersas y características locales.
    *   **Aproximadores Universales de Funciones**: Al igual que las redes de alimentación directa, las redes RBF son aproximadores universales de funciones.

*   **Entrenamiento de una Red RBF**
    El entrenamiento de una red RBF es un proceso en dos etapas, lo que la diferencia de las redes de alimentación directa.
    *   **Entrenamiento de la Capa Oculta**:
        *   Generalmente se realiza de **manera no supervisada**.
        *   **Parámetros**: Los vectores prototipo (µi) y los anchos de banda (σi). Comúnmente, todas las unidades comparten el mismo ancho de banda (σ), aunque los prototipos son diferentes.
        *   **Selección de Prototipos**:
            *   **Muestreo aleatorio**: Se seleccionan `m` puntos de entrenamiento aleatorios como prototipos. Problema: sobre-representación de regiones densas y poca representación de regiones dispersas.
            *   **Algoritmo k-means**: La elección más común. Los centroides de `m` clústeres creados por k-means se usan como prototipos.
            *   **Variantes de algoritmos de agrupación**: Como el uso de árboles de decisión.
            *   **Algoritmo de Mínimos Cuadrados Ortogonales (OLS)**: Selecciona prototipos uno por uno de los datos de entrenamiento para minimizar el error de predicción en un conjunto de prueba de validación. Este método introduce cierto nivel de supervisión.
        *   **Selección del Ancho de Banda (σ)**:
            *   **Heurísticas**: `σ = dmax / √m` o `σ = 2 · dave`, donde `dmax` es la distancia máxima y `dave` la distancia promedio entre los centros prototipo.
            *   **Basado en vecinos cercanos**: `σi` puede ser la distancia del i-ésimo prototipo a su r-ésimo vecino más cercano entre los prototipos.
            *   **Ajuste fino (fine-tuning)**: Se prueban valores candidatos de σ en un conjunto de datos de validación para encontrar el que minimiza el error. Esto introduce una supervisión "suave" pero evita mínimos locales.
        *   **Limitaciones del Entrenamiento Totalmente Supervisado de la Capa Oculta**: Es posible entrenar los prototipos y anchos de banda con retropropagación (backpropagation), pero el problema es que la superficie de pérdida de las RBFs tiene muchos mínimos locales, y el entrenamiento supervisado tiende a quedarse atrapado en ellos. Además, la ventaja de eficiencia de las RBFs se pierde con la retropropagación completa.
    *   **Entrenamiento de la Capa de Salida**:
        *   Se realiza de **manera supervisada** después de entrenar la capa oculta.
        *   **Objetivo**: Aprender el vector de pesos `W = [w1 ... wm]`.
        *   **Pérdida de Mínimos Cuadrados (Regresión)**: Para objetivos de valor real, la función de pérdida es `L = 1/2 ||HWᵀ - y||²`. Con regularización de Tikhonov (`L = 1/2 ||HWᵀ - y||² + λ/2 ||W||²`), la solución para `Wᵀ` se puede encontrar en forma cerrada: `Wᵀ = (HᵀH + λI)⁻¹Hᵀy`.
        *   **Limitaciones de la forma cerrada**: La matriz `HᵀH` puede ser muy grande (m x m), haciendo la inversión inviable en la práctica (ej. en métodos kernel donde `m=n`). Por ello, se usa **descenso de gradiente estocástico (SGD)** o descenso de gradiente por mini-lotes para actualizar `W`.
        *   **Pérdidas para Clasificación**:
            *   **Criterio del Perceptrón**: `L = max{-yi(W ·Hi), 0}`.
            *   **Hinge Loss**: `L = max{1− yi(W ·Hi), 0}`. Usada frecuentemente en máquinas de vectores de soporte (SVM).
            *   También se menciona la pérdida logística (logistic loss).
        *   Para variables objetivo binarias, se pueden tratar como respuestas numéricas en {−1,+1} y usar el enfoque de mínimos cuadrados, lo que es equivalente al discriminante de Fisher o al método de Widrow-Hoff en un espacio de mayor dimensionalidad.

*   **Relación con Métodos Kernel**
    *   Las redes RBF son **generalizaciones directas** de la clase de métodos kernel, como la regresión kernel y las máquinas de vectores de soporte kernel (SVM kernel).
    *   Se puede demostrar que las RBFs pueden simular casi cualquier método kernel cambiando la función de pérdida.
    *   **Casos Especiales**:
        *   **Regresión Kernel**: Una RBF network se reduce a regresión kernel cuando los prototipos (`µj`) se establecen como los propios puntos de entrenamiento (`Xj`), y todos los anchos de banda (`σ`) son iguales.
        *   **SVM Kernel**: De manera similar, una SVM kernel es un caso especial de las RBFs cuando los prototipos se establecen como los puntos de entrenamiento y se utiliza la función de pérdida `hinge`.
    *   **Flexibilidad Superior**: Las redes RBF son más potentes y flexibles que los métodos kernel tradicionales porque permiten elegir libremente el número de nodos ocultos y los prototipos, lo que puede mejorar la precisión y la eficiencia.

*   **Aplicaciones**
    *   **Clasificación**: Transforman los datos para promover la separabilidad lineal, permitiendo que clasificadores lineales como el perceptrón o la SVM funcionen eficazmente.
    *   **Regresión**: Predicción de valores continuos.
    *   **Interpolación**: Una de las primeras aplicaciones. Aquí, cada punto de entrenamiento es un prototipo (`m=n`), y la red puede encontrar una solución exacta con error cero, comportándose como una regresión lineal con una matriz `H` cuadrada e invertible.

*   **Ventajas y Limitaciones**
    *   **Robustez al Ruido**: La creación no supervisada de la capa oculta las hace robustas a diversos tipos de ruido, incluyendo el ruido adversario, una propiedad compartida con las SVM.
    *   **Eficiencia de Entrenamiento**: La separación del entrenamiento de la capa oculta y de salida contribuye a la eficiencia, especialmente con métodos no supervisados o semi-supervisados para la capa oculta.
    *   **Limitaciones en el Aprendizaje de Estructuras Complejas**: La única capa oculta de una red RBF limita la cantidad de estructura que se puede aprender en comparación con las redes de alimentación directa profundas, que son efectivas para datos con estructuras ricas.
    *   **Riesgo de Overfitting**: Aumentar el número de unidades ocultas o usar anchos de banda muy pequeños puede llevar a overfitting, especialmente con conjuntos de datos pequeños.

*   **Uso Actual**
    Las redes RBF han sido menos utilizadas en los últimos años y se consideran una categoría "olvidada" de arquitecturas neuronales. Sin embargo, tienen un potencial significativo en escenarios donde se utilizan métodos kernel.

En resumen, las Redes de Funciones de Base Radial ofrecen una arquitectura única y poderosa que transforma datos no lineales en espacios linealmente separables, a menudo de forma eficiente mediante un entrenamiento híbrido (no supervisado para la capa oculta y supervisado para la capa de salida). Su estrecha relación con los métodos kernel las posiciona como una alternativa flexible y generalizada a estos, con aplicaciones en clasificación, regresión e interpolación.

***

# Parte VI

***

# Máquinas de Boltzmann Restringidas (RBM)

Las Máquinas de Boltzmann Restringidas (RBM) son un tipo de **modelo fundamentalmente diferente** a las redes neuronales de alimentación directa (feed-forward) convencionales. A diferencia de estas últimas, que mapean entradas a salidas, las RBM **aprenden los estados probabilísticos de una red para un conjunto de entradas**, lo que las hace útiles para el modelado no supervisado.


### 1. Concepto y Características Fundamentales
*   Las RBM son redes que modelan la **distribución de probabilidad conjunta de atributos observados y atributos ocultos**.
*   Son **modelos no supervisados** que generan **representaciones latentes de características** de los puntos de datos.
*   A diferencia de la mayoría de los autoencoders (excepto el autoencoder variacional) que crean representaciones ocultas deterministas, las RBM crean una **representación oculta estocástica** de cada punto, lo que requiere una forma de entrenamiento y uso fundamentalmente diferente.
*   Son **redes no dirigidas** porque están diseñadas para aprender relaciones probabilísticas, no mapeos de entrada-salida.
*   Son un caso especial de modelos gráficos probabilísticos conocidos como **campos aleatorios de Markov**.

### 2. Perspectiva Histórica
*   Las RBM han evolucionado a partir del **modelo clásico de red de Hopfield**.
    *   Una red de Hopfield contiene nodos con estados binarios y crea un modelo determinista de las relaciones entre atributos usando bordes ponderados.
    *   La **función objetivo de una red de Hopfield se conoce como función de energía**, y su minimización fomenta que los nodos con pesos positivos tengan estados similares y los nodos con pesos negativos tengan estados diferentes.
    *   Las redes de Hopfield son **memoria asociativa**: dada una entrada, iterativamente voltean bits para mejorar la función objetivo hasta encontrar un mínimo local, que a menudo es un punto memorizado de los datos de entrenamiento.
    *   Se entrenan con la **regla de aprendizaje de Hebbian**, que fortalece las conexiones entre neuronas con salidas correlacionadas.
    *   Las redes de Hopfield tienen una **capacidad de almacenamiento limitada** (aproximadamente 0.15 * d ejemplos de entrenamiento para d unidades visibles), lo que las hace ineficientes para el almacenamiento de bits en comparación con la cantidad de pesos que poseen.
    *   Su poder expresivo puede aumentarse añadiendo **unidades ocultas**, que capturan la estructura latente de los datos.
*   La red de Hopfield evolucionó hacia la **Máquina de Boltzmann**, que utiliza **estados probabilísticos** (distribuciones de Bernoulli) para representar atributos binarios.
    *   Las Máquinas de Boltzmann contienen tanto **estados visibles** (para datos observados) como **estados ocultos** (para variables latentes).
    *   La **principal diferencia entre la Máquina de Boltzmann y la Máquina de Boltzmann Restringida es que esta última solo permite conexiones entre unidades ocultas y visibles**, lo que simplifica los algoritmos de entrenamiento.
*   Inicialmente, las RBM eran consideradas lentas de entrenar, pero ganaron popularidad con algoritmos más rápidos a principios de siglo y fueron un componente clave en el concurso Netflix Prize.

### 3. Entrenamiento de las RBM
*   El objetivo es **maximizar la verosimilitud logarítmica** del conjunto de datos de entrenamiento.
*   La probabilidad de un estado en una RBM (similar a una Máquina de Boltzmann) se define aplicando la función sigmoide a la brecha de energía (diferencia de energía entre configuraciones).
*   **Generación de datos**: Debido a las dependencias circulares, la generación de datos se realiza mediante un proceso iterativo de muestreo de estados usando distribuciones condicionales hasta alcanzar el **equilibrio térmico**, un proceso conocido como **muestreo de Gibbs** o **Markov Chain Monte Carlo (MCMC)**.
*   **Aprendizaje de pesos**:
    *   La regla de actualización de pesos implica la **diferencia entre las correlaciones de estados observadas en los datos y las correlaciones de estados del modelo**.
    *   Se requieren dos tipos de muestras: **muestras centradas en los datos** (estados visibles fijados a datos de entrenamiento, estados ocultos muestreados) y **muestras del modelo** (todos los estados muestreados libremente).
    *   El proceso MCMC para la Máquina de Boltzmann es lento en la práctica.
*   **Algoritmo de Divergencia Contrastiva (CDk)**:
    *   Es una **aproximación más rápida** para entrenar RBMs.
    *   La variante más rápida, **CD1**, utiliza solo una iteración adicional de muestreo Monte Carlo para generar las muestras negativas.
    *   Un mayor número de iteraciones (CDk con k>1) mejora la precisión del gradiente a expensas de la velocidad.
    *   Una estrategia común es **aumentar progresivamente el valor de k** a medida que el descenso de gradiente se acerca a una mejor solución.
*   **Problemas prácticos e improvisaciones**:
    *   A menudo, se utilizan **valores de probabilidad** (activaciones sigmoides) en lugar de valores binarios muestreados para reducir el ruido durante el entrenamiento, aunque esto sea teóricamente incorrecto.
    *   Los **pesos se inicializan** con una distribución Gaussiana de media cero y desviación estándar de 0.01.
    *   Los **sesgos visibles** se inicializan a `log(pi/(1-pi))` (donde pi es la fracción de puntos con el valor 1), y los **sesgos ocultos** a 0.
    *   El tamaño del **mini-lote** debe estar entre 10 y 100, y el orden de los ejemplos debe aleatorizarse.

### 4. Aplicaciones de las RBM
*   **Reducción de Dimensionalidad y Reconstrucción de Datos**:
    *   Las unidades ocultas de una RBM contienen una **representación reducida de los datos**.
    *   Una RBM puede "desplegarse" para crear un **modelo dirigido** que funcione como un autoencoder virtual, con porciones de codificador y decodificador.
    *   Los pesos entrenados en la RBM pueden usarse para inicializar y afinar con retropropagación una red neuronal tradicional.
    *   Esta capacidad de inicialización es una de las **contribuciones históricas importantes de las RBM al pre-entrenamiento**.
*   **Filtrado Colaborativo**:
    *   Se utilizan para matrices de calificaciones incompletamente especificadas, un problema común en los sistemas de recomendación.
    *   Para manejar calificaciones no binarias (ej. 1 a 5 estrellas), las unidades visibles se modelan como **unidades softmax de 5 vías** (codificación one-hot de la calificación).
    *   Se define una RBM por cada usuario, compartiendo los pesos entre ellas.
    *   Una vez entrenadas, las RBMs pueden hacer predicciones usando activaciones de valor real (probabilidades) en lugar de estados binarios, lo que las convierte en una red neuronal con unidades logísticas y softmax.
    *   Se puede aplicar una técnica de **regularización mediante factorización condicional** para reducir el número de parámetros en matrices de pesos grandes, asumiendo una estructura de bajo rango (W = UVᵀ).
*   **Clasificación**:
    *   La forma más común es como un **procedimiento de pre-entrenamiento**: la RBM se usa para la ingeniería de características no supervisada, luego se "desenrolla" en una arquitectura de codificador-decodificador y se afina con retropropagación.
    *   Alternativamente, la clasificación se puede ver como un caso simplificado de **completado de matriz**, donde la columna de clase es el valor faltante a predecir.
    *   La capa visible contiene nodos para características binarias y unidades softmax para la etiqueta de clase (codificación one-hot).
    *   Se puede usar un **enfoque discriminativo** para entrenar la RBM, maximizando la verosimilitud condicional de la clase verdadera, en lugar de un modelo generativo que no optimiza completamente la precisión de la clasificación.
*   **Modelos de Temas (Topic Models)**:
    *   Forma de reducción de dimensionalidad para datos de texto.
    *   Similar al filtrado colaborativo, se crea una RBM para cada documento, con unidades visibles agrupadas por palabras (softmax) y unidades ocultas representando temas.
    *   Las unidades visibles comparten el mismo conjunto de parámetros.
*   **Aprendizaje con Datos Multimodales**:
    *   Las RBM pueden crear una **representación latente compartida** para datos de múltiples modalidades (ej., imagen y texto).
    *   Esta arquitectura es similar a la de clasificación, mapeando dos tipos de características a un conjunto de estados ocultos compartidos.

### 5. Uso de RBM más allá de Datos Binarios
*   Para **datos categóricos u ordinales** (ej., calificaciones, conteos de palabras), se usa el **enfoque softmax** para las unidades visibles.
*   Para **datos de valor real**, se pueden usar **unidades visibles Gaussianas** y **unidades ocultas de ReLU**.
    *   El uso de unidades Gaussianas puede ser inestable respecto a la varianza `σ`, y a menudo se normalizan los datos de entrada a varianza unitaria.
    *   Las unidades ReLU se modifican con ruido gaussiano para codificar más información que las unidades binarias.

### 6. Apilamiento de Máquinas de Boltzmann Restringidas (Redes Profundas)
*   Las RBM son adecuadas para crear **redes profundas** y fueron usadas antes que las redes neuronales convencionales para este propósito mediante el **pre-entrenamiento**.
*   El entrenamiento se realiza **secuencialmente**:
    1.  Se entrena la primera RBM usando los datos de entrenamiento para las unidades visibles.
    2.  Las salidas (representaciones ocultas) de la primera RBM se copian y se usan como entradas para entrenar la segunda RBM.
    3.  Este proceso se repite para cada capa deseada.
*   Una vez entrenados los pesos de las RBM individuales, se pueden ensamblar en una **red codificador-decodificador dirigida** con activaciones continuas (sigmoide).
*   Aunque las RBM son modelos simétricos y discretos, sus pesos aprendidos son una **excelente inicialización** para una red neuronal tradicional que luego puede ser **afinada con retropropagación** para mejorar el aprendizaje (particularmente crucial para el aprendizaje supervisado).
*   Las **Máquinas de Boltzmann Profundas (DBM)** y las **Redes de Creencia Profundas (DBN)** son variaciones de RBM apiladas con interacciones bidireccionales o combinaciones de capas unidireccionales y bidireccionales.

En resumen, las RBM son **modelos probabilísticos no supervisados** que generan representaciones latentes, con un papel histórico crucial en la popularización del pre-entrenamiento para redes profundas. Aunque naturalmente trabajan con estados binarios, se han extendido a otros tipos de datos (categóricos, ordinales, reales) mediante adaptaciones en su función de energía y reglas de muestreo. Su entrenamiento, aunque desafiante por la naturaleza estocástica, se ha optimizado con algoritmos como la divergencia contrastiva, permitiendo su aplicación en diversas áreas como la reducción de dimensionalidad, el filtrado colaborativo, la clasificación y el modelado de temas.

***

# Parte VII

***

# Redes Neuronales Recurrentes: Fundamentos y Aplicaciones

Las Redes Neuronales Recurrentes (RNN, por sus siglas en inglés) son una clase de arquitecturas de redes neuronales intrínsecamente diseñadas para manejar datos secuenciales, a diferencia de las arquitecturas tradicionales que asumen que los atributos son en gran medida independientes entre sí. Este enfoque es crucial para tipos de datos como series de tiempo, texto y datos biológicos, donde existen dependencias secuenciales significativas entre los atributos.

### Necesidad y Desafíos de los Datos Secuenciales

En los datos de series de tiempo, los valores en estampas de tiempo sucesivas están estrechamente relacionados. Si estos valores se trataran como características independientes, se perdería información clave sobre sus relaciones. De manera similar, aunque el texto a menudo se procesa como una "bolsa de palabras", se obtienen mejores conocimientos semánticos al considerar el orden de las palabras, lo que hace que los modelos que tienen en cuenta la secuencia sean importantes. Los datos biológicos también contienen secuencias, donde los símbolos pueden corresponder a aminoácidos o nucleobases que forman el ADN.

Las redes neuronales convencionales enfrentan desafíos importantes al procesar datos secuenciales debido a la longitud variable de las entradas y la falta de información sobre las dependencias secuenciales entre elementos sucesivos. Por ejemplo, para el análisis de sentimientos, una red neuronal tradicional con un número fijo de entradas no podría manejar oraciones de diferentes longitudes, lo que resultaría en entradas faltantes o excedentes. Además, pequeños cambios en el orden de las palabras pueden alterar drásticamente el significado semántico de una oración, y las redes convencionales no codifican esta información de orden de manera directa.

Los dos requisitos principales para el procesamiento de secuencias son:
1.  La capacidad de recibir y procesar entradas en el mismo orden en que están presentes en la secuencia.
2.  El tratamiento de las entradas en cada estampa de tiempo de manera similar en relación con el historial previo de entradas.

Un desafío clave es construir una red neuronal con un número fijo de parámetros, pero con la capacidad de procesar un número variable de entradas.

### La Arquitectura de las Redes Neuronales Recurrentes

Las RNN satisfacen estas necesidades de forma natural. En una RNN, existe una correspondencia uno a uno entre las capas de la red y las posiciones específicas en la secuencia (también llamadas "estampas de tiempo"). En lugar de un número variable de entradas en una sola capa de entrada, la red contiene un número variable de capas, y cada capa tiene una única entrada correspondiente a esa estampa de tiempo. Las entradas pueden interactuar directamente con las capas ocultas posteriores según sus posiciones en la secuencia.

Una característica distintiva de las RNN es que cada capa utiliza el mismo conjunto de parámetros, asegurando un modelado similar en cada estampa de tiempo y, por lo tanto, manteniendo fijo el número de parámetros. En otras palabras, la misma arquitectura de capa se repite en el tiempo, lo que da a la red su nombre "recurrente". Las RNN también pueden verse como redes de alimentación hacia adelante (feed-forward) con una estructura específica basada en la noción de capas temporales, lo que les permite tomar una secuencia de entradas y producir una secuencia de salidas. Cada capa temporal puede recibir un punto de datos de entrada (sea un atributo único o múltiples atributos) y opcionalmente producir una salida multidimensional.

El núcleo de una RNN se ilustra con la presencia de un auto-bucle en su representación simplificada (Figura 7.2(a) en la fuente), que hace que el estado oculto de la red cambie después de la entrada de cada palabra en la secuencia. En la práctica, este bucle se "despliega" en una red "en capas temporales" que se asemeja más a una red de alimentación hacia adelante (Figura 7.2(b)). Esta representación es matemáticamente equivalente pero más fácil de comprender. Las matrices de pesos (Wxh, Whh, Why) se comparten entre las diferentes capas temporales para garantizar que se utilice la misma función en cada estampa de tiempo.

Los valores individuales en una secuencia pueden ser reales o simbólicos. Las RNN pueden usarse para ambos tipos, aunque el uso de valores simbólicos es más común, especialmente en datos de texto. El texto se asume como la entrada predeterminada, donde los símbolos corresponden a identificadores de palabras del léxico, aunque también se consideran elementos como caracteres o valores reales.

En la estampa de tiempo `t`, la entrada es `xt`, el estado oculto es `ht`, y la salida es `yt`. `xt` y `yt` son `d`-dimensionales (para un léxico de tamaño `d`), y `ht` es `p`-dimensional, donde `p` regula la complejidad de la incrustación. El estado oculto `ht` se calcula como una función del vector de entrada `xt` en el tiempo `t` y del vector oculto `ht-1` del tiempo `t-1`: `ht = f(ht-1, xt)`. Una función separada `yt = g(ht)` se usa para aprender las probabilidades de salida desde los estados ocultos. Las mismas matrices de pesos (Wxh, Whh, Why) y funciones de activación (como tanh para la capa oculta y softmax para la salida) se utilizan en cada estampa de tiempo. La naturaleza recursiva de la Ecuación 7.1 permite a la red calcular una función de entradas de longitud variable.

### Modelado de Lenguaje con RNN

Un uso clásico de las RNN es el modelado de lenguaje, donde la red predice la siguiente palabra dada la historia previa de palabras. Por ejemplo, al ingresar la palabra "El" de la frase "El gato persiguió al ratón", la salida sería un vector de probabilidades para la siguiente palabra, que idealmente incluiría "gato". El vector de salida `yt` son valores continuos que se convierten a probabilidades usando una función softmax. Aunque los errores son comunes en las primeras iteraciones, la red mejora con el entrenamiento repetido.

Una vez entrenado, el modelo de lenguaje se puede usar para generar muestras de lenguaje arbitrarias. Esto se logra tomando el token `<START>` como entrada para generar las probabilidades del primer token, luego muestreando uno de los tokens generados (basado en la probabilidad predicha) y usándolo como entrada para la siguiente estampa de tiempo. Para mejorar la precisión, se puede usar la búsqueda por haces (beam search). Si se predice el token `<END>`, se indica el final del segmento de texto. Los textos generados pueden ser sintácticamente correctos, pero a menudo carecen de significado semántico, especialmente en las primeras iteraciones de entrenamiento. Sin embargo, al continuar el entrenamiento o al condicionar la salida a una entrada contextual adicional (como la representación neural de una imagen), las RNN pueden generar salidas inteligentes, como descripciones gramaticalmente correctas de imágenes.

### Entrenamiento de RNN: Retropropagación a Través del Tiempo (BPTT)

El entrenamiento de las RNN implica la minimización de una función de pérdida, típicamente la suma de los logaritmos negativos de las probabilidades softmax de las palabras correctas en cada estampa de tiempo. La derivada de la función de pérdida con respecto a las salidas sin procesar se calcula y se utiliza en el proceso de retropropagación.

El principal desafío en el entrenamiento es el manejo de los pesos compartidos entre las diferentes capas temporales. La retropropagación a través del tiempo (BPTT) aborda esto:
1.  Se realiza una propagación hacia adelante de la entrada secuencialmente para calcular los errores en cada estampa de tiempo.
2.  Se calculan los gradientes de los pesos en la dirección inversa en la red desplegada, asumiendo que los pesos en diferentes capas de tiempo son distintos.
3.  Finalmente, las contribuciones de los gradientes de las diferentes "instanciaciones temporales" de los parámetros de peso se suman para crear una actualización unificada para cada parámetro de peso compartido.

**Retropropagación Truncada a Través del Tiempo**: Para secuencias muy largas, la profundidad de la red puede ser enorme, lo que lleva a problemas computacionales, de convergencia y de uso de memoria. La BPTT truncada resuelve esto realizando actualizaciones de retropropagación solo sobre segmentos de la secuencia de longitud modesta (por ejemplo, 100). La propagación hacia adelante mantiene los valores de estado correctos, y los valores de la capa final de un segmento se usan para calcular los estados del siguiente segmento, asegurando que el estado siempre se mantenga con precisión.

**Cuestiones Prácticas**:
*   **Inicialización de pesos**: Las entradas de cada matriz de pesos se inicializan con valores pequeños. También se puede preentrenar la matriz de entrada Wxh con incrustaciones word2vec, lo que es útil cuando los datos de entrenamiento son escasos.
*   **Tokens de inicio/fin**: Los datos de entrenamiento a menudo incluyen tokens especiales como `<START>` y `<END>` para ayudar al modelo a reconocer unidades de texto.

### Desafíos del Entrenamiento: Problemas de Gradiente Desvaneciente y Explosivo

Las RNN son notoriamente difíciles de entrenar porque la red con capas de tiempo es muy profunda, especialmente con secuencias de entrada largas. Esto, combinado con el uso de parámetros compartidos, puede generar inestabilidades. El problema principal es el de los gradientes desvanecientes y explosivos.

*   **Gradiente Desvaneciente**: Ocurre cuando los gradientes se vuelven extremadamente pequeños a medida que se retropropagan a través de muchas capas, lo que dificulta el aprendizaje de las dependencias a largo plazo. Las funciones de activación como `tanh` (cuya derivada es casi siempre menor que 1) pueden exacerbar este problema.
*   **Gradiente Explosivo**: Ocurre cuando los gradientes se vuelven extremadamente grandes, lo que lleva a actualizaciones de pesos inestables y divergencia del modelo.

Cuando esto ocurre, la elección del tamaño de paso del descenso de gradiente se vuelve crítica: un tamaño muy pequeño causa poco progreso, mientras que uno muy grande puede hacer que el paso sobrepase el punto óptimo de manera inestable. Las superficies de pérdida en las RNN a menudo presentan "acantilados" donde los gradientes cambian drásticamente.

**Soluciones a los Problemas de Gradiente**:
*   **Regularización Fuerte**: Reduce la inestabilidad, pero puede limitar el potencial del modelo.
*   **Recorte de Gradiente (Gradient Clipping)**: Efectivo para el problema del gradiente explosivo. Puede ser basado en valor (recorta los componentes más grandes del gradiente) o basado en la norma (re-escala el vector de gradiente si su norma excede un umbral).
*   **Gradientes de Orden Superior**: Computacionalmente costosos, pero métodos libres de Hessian han mostrado éxito.
*   **Inicialización y Métodos de Momento**: Una buena inicialización de los pesos y el uso de métodos de momento pueden ayudar a evitar la inestabilidad.
*   **Normalización por Capas (Layer Normalization)**: Aunque la normalización por lotes (batch normalization) se puede adaptar, la normalización por capas es más efectiva para las RNN. La normalización por lotes presenta desafíos en las RNN porque las estadísticas de lotes varían con las capas de tiempo y pueden no estar disponibles para secuencias de prueba más largas. La normalización por capas, en cambio, se realiza sobre una única instancia de entrenamiento, utilizando todas las activaciones actuales en esa capa. Esto asegura que las magnitudes de las activaciones no aumenten o disminuyan continuamente, permitiendo flexibilidad a través de parámetros aprendibles.
*   **Variantes de RNN**: Específicamente, las redes de memoria a corto y largo plazo (LSTM) y las unidades recurrentes gating (GRU) abordan intrínsecamente estos problemas.

### Variantes y Mejoras de las RNN

Las RNN se han mejorado con varias arquitecturas para superar las limitaciones y abordar problemas de gradiente:

1.  **Redes Recurrentes Bidireccionales (Bi-RNN)**:
    *   Una desventaja de las RNN unidireccionales es que el estado en una unidad de tiempo solo tiene conocimiento de las entradas pasadas, no de los estados futuros.
    *   Las Bi-RNN tienen estados ocultos separados (`h(f)t` para adelante y `h(b)t` para atrás) que interactúan solo entre sí en sus respectivas direcciones. Ambos reciben entrada del mismo vector `xt` y interactúan con el mismo vector de salida `yt`.
    *   Son muy útiles en aplicaciones donde el contexto de ambos lados de un elemento es beneficioso, como el reconocimiento de escritura a mano, el reconocimiento de voz o el etiquetado de partes del discurso en una oración.
    *   Tienen matrices de parámetros separadas para las direcciones hacia adelante y hacia atrás.
    *   El algoritmo BPTT se modifica para las Bi-RNN: se calculan los estados ocultos hacia adelante y hacia atrás de forma independiente, luego los estados de salida, y finalmente se realizan las retropropagaciones de gradientes para los estados hacia adelante y hacia atrás de forma independiente antes de agregarlos para los parámetros compartidos.

2.  **Redes Recurrentes Multicapa**:
    *   En aplicaciones prácticas, se utiliza una arquitectura multicapa para construir modelos de mayor complejidad.
    *   Los nodos en capas de nivel superior reciben entrada de los de capas de nivel inferior.
    *   Los pesos se comparten entre diferentes estampas de tiempo, pero no entre diferentes capas.
    *   Comúnmente se usan dos o tres capas; un mayor número de capas requiere más datos de entrenamiento para evitar el sobreajuste. Estas arquitecturas se combinan a menudo con variantes avanzadas como LSTM o GRU.

3.  **Redes de Estado de Eco (Echo-State Networks, ESN)**:
    *   Una simplificación de las RNN que funcionan bien cuando la dimensionalidad de la entrada es pequeña.
    *   Son útiles para el modelado de regresión de series de tiempo de valores reales individuales o pequeños conjuntos, especialmente en horizontes de tiempo largos. Sin embargo, no son adecuadas para el texto debido a la alta dimensionalidad de entrada.
    *   Utilizan pesos aleatorios en la capa oculta a oculta e incluso en la capa de entrada a oculta; solo la capa de salida se entrena, típicamente de forma lineal y sin necesidad de retropropagación, lo que hace que el entrenamiento sea muy rápido.
    *   Una salvedad importante es que el valor propio (eigenvector) más grande de la matriz de pesos `Whh` debe establecerse en 1, lo que se logra escalando sus entradas aleatorias.
    *   Su utilidad principal es para la inicialización de pesos en RNN más complejas, sirviendo como una forma ligera de preentrenamiento.

4.  **Memoria a Corto y Largo Plazo (Long Short-Term Memory, LSTM)**:
    *   Propuesta para abordar los problemas de gradiente desvaneciente y explosivo.
    *   Introducen un vector oculto adicional, `c(k)t`, llamado "estado de celda", que actúa como una memoria a largo plazo.
    *   La LSTM controla el flujo de información a través de la memoria a largo plazo mediante "compuertas":
        *   **Compuerta de Entrada (`i`)**: Decide qué nueva información se añadirá al estado de la celda.
        *   **Compuerta de Olvido (`f`)**: Decide qué información del estado de la celda anterior (`c(k)t-1`) debe ser "olvidada" o reseteada.
        *   **Compuerta de Salida (`o`)**: Decide cuánto del estado de la celda (`c(k)t`) se "fuga" al estado oculto (`h(k)t`) actual.
    *   Estas actualizaciones del estado de la celda son de forma aditiva (`c(k)t = f ⋅ c(k)t-1 + i ⋅ c`), lo que ayuda a evitar el problema del gradiente desvaneciente causado por las actualizaciones multiplicativas.
    *   Las LSTM actúan como "superautopistas de gradiente", permitiendo un flujo de gradiente más estable y persistente, lo que las hace adecuadas para modelar dependencias de largo alcance en secuencias.

5.  **Unidades Recurrentes Gating (Gated Recurrent Units, GRU)**:
    *   Pueden verse como una simplificación de la LSTM que no utiliza estados de celda explícitos.
    *   Utiliza dos compuertas principales:
        *   **Compuerta de Actualización (`z`)**: Decide la fuerza relativa de la contribución de una actualización basada en la matriz y una contribución más directa del vector oculto anterior (`h(k)t-1`). Actúa simultáneamente como las compuertas de entrada y olvido de la LSTM (a través de `z` y `1-z`).
        *   **Compuerta de Reseteo (`r`)**: Decide cuánto del estado oculto anterior se "olvida" antes de combinarse con la entrada actual.
    *   Al permitir una copia directa (parcial) de los estados ocultos de la capa anterior, el flujo de gradiente se vuelve más estable durante la retropropagación.
    *   Las GRU son más simples y eficientes que las LSTM, con menos parámetros. El rendimiento relativo entre GRU y LSTM depende de la tarea; las GRU pueden generalizar mejor con menos datos, mientras que las LSTM son preferibles para secuencias más largas y conjuntos de datos más grandes.

### Aplicaciones de las Redes Neuronales Recurrentes

Las RNN y sus variantes tienen numerosas aplicaciones en el aprendizaje automático, principalmente en el procesamiento de información secuencial:

1.  **Modelado de Lenguaje Condicional**: La salida de una RNN es un modelo de lenguaje que se mejora con el contexto de otra red neural.
    *   **Subtitulado Automático de Imágenes (Image Captioning)**: Entrenada con pares imagen-subtítulo. Una red neuronal convolucional (CNN) genera una representación vectorial de la imagen (`v`), que se alimenta a la RNN (generalmente solo en la primera estampa de tiempo). La CNN y la RNN se entrenan conjuntamente para predecir un subtítulo relevante.
    *   **Aprendizaje Secuencia-a-Secuencia y Traducción Automática**: Dos RNN se conectan de extremo a extremo. La primera RNN (codificador, RNN1) procesa la sentencia en el lenguaje de origen, acumulando conocimiento en su estado oculto sin producir salidas hasta el símbolo de fin de sentencia. La segunda RNN (decodificador, RNN2) usa esta codificación para generar la sentencia en el lenguaje de destino, palabra por palabra. Ambas redes se entrenan conjuntamente. Una debilidad es que tienden a funcionar mal con oraciones largas, una solución es invertir el orden de entrada de la sentencia de origen.

2.  **Aprovechamiento de Salidas Específicas de Tokens**: Las salidas en diferentes tokens se utilizan para aprender propiedades distintas a un modelo de lenguaje.
    *   **Sistemas de Preguntas y Respuestas (QA Systems)**: Una aplicación natural del aprendizaje secuencia-a-secuencia. Pueden inferir respuestas directamente de frases en preguntas o transformar preguntas en consultas a bases de conocimiento estructuradas. Requieren una mayor capacidad de razonamiento y comprensión de relaciones entre entidades. Los Memory Networks son una arquitectura que se adapta bien a estos escenarios.
    *   **Clasificación a Nivel de Sentencia**: Cada sentencia se trata como una instancia de entrenamiento/prueba para clasificación, por ejemplo, en el análisis de sentimientos. Se predice una única etiqueta de clase al final de la sentencia. Las RNN pueden manejar "cambiadores de valencia contextuales" (como "no" antes de "amor"), que cambian el sentimiento de la frase.
    *   **Clasificación a Nivel de Token con Características Lingüísticas**: Aplicaciones como la extracción de información (identificar personas, lugares, organizaciones) y la segmentación de texto. La entrada en cada estampa de tiempo incluye la codificación one-hot del token y características lingüísticas adicionales (como la capitalización, parte del discurso). A menudo se utilizan Bi-RNN para beneficiarse del contexto de ambos lados.
    *   **Pronóstico y Predicción de Series de Tiempo**: Elección natural para datos de series de tiempo, donde las entradas son vectores de valores reales. Las ESN son particularmente efectivas para un número pequeño de series de tiempo, superando a los modelos autorregresivos tradicionales mediante la introducción de no linealidades y la expansión aleatoria del espacio de características.
    *   **Sistemas de Recomendación Temporal**: Aprovechan las estampas de tiempo asociadas con las calificaciones de los usuarios. Predicen una calificación basada en características estáticas del artículo, características estáticas del usuario y características dinámicas del usuario (historial de accesos cambiante), donde la RNN modela estas últimas.
    *   **Predicción de Estructura Secundaria de Proteínas**: Los elementos de la secuencia son aminoácidos. Se reduce a un problema de clasificación a nivel de token, donde cada posición se clasifica como hélice alfa, lámina beta o espiral. Las Bi-RNN son apropiadas aquí debido al beneficio del contexto en ambos lados.
    *   **Reconocimiento de Voz de Extremo a Extremo**: Transcribe archivos de audio crudos a secuencias de caracteres. Las Bi-RNN son las más adecuadas. Un desafío es la alineación entre la representación de la trama de audio y la secuencia de transcripción (paradoja de Sayre), resuelto con la Clasificación Temporal Conectada (CTC) y programación dinámica.
    *   **Reconocimiento de Escritura a Mano**: Similar al reconocimiento de voz. La entrada es una secuencia de coordenadas (x,y) de la punta del lápiz, y la salida es una secuencia de caracteres escritos. También enfrenta la paradoja de Sayre, y la CTC se utiliza para resolverla.

En resumen, las RNN son una herramienta poderosa para el modelado de secuencias, capaces de capturar dependencias temporales y contextuales que las redes neuronales tradicionales no pueden. A pesar de los desafíos de entrenamiento como los problemas de gradiente, las mejoras arquitectónicas como LSTM, GRU y Bi-RNN, junto con técnicas de normalización, han hecho de las RNN una solución fundamental para una amplia gama de aplicaciones en procesamiento de lenguaje natural, visión por computadora y bioinformática.

***

# Parte VIII

***

# Redes Neuronales Convolucionales (CNN)

Las Redes Neuronales Convolucionales (CNNs) son una clase de redes neuronales profundas diseñadas específicamente para trabajar con entradas estructuradas en forma de cuadrícula que exhiben fuertes dependencias espaciales en regiones locales. El ejemplo más común de datos estructurados en cuadrícula son las imágenes bidimensionales, donde las ubicaciones espaciales adyacentes a menudo tienen valores de color de píxeles similares. Las CNNs pueden manejar entradas tridimensionales, donde una dimensión adicional captura diferentes colores, creando un volumen de entrada tridimensional. Aunque se utilizan predominantemente para datos de imágenes, también pueden aplicarse a datos temporales, espaciales y espaciotemporales, como texto o series de tiempo.

Una propiedad importante de los datos de imagen, que las CNNs aprovechan, es la invarianza de traslación; por ejemplo, una banana se interpreta igual independientemente de su posición en una imagen. Las CNNs tienden a generar valores de características similares a partir de regiones locales con patrones parecidos.

**Característica Definitoria: La Operación de Convolución**
La característica principal que define a las CNNs es la operación de convolución. Esta operación es un producto punto entre un conjunto de pesos estructurados en cuadrícula (conocidos como filtros o kernels) y entradas estructuradas de manera similar tomadas de diferentes localidades espaciales en el volumen de entrada. La operación de convolución es particularmente útil para datos con un alto nivel de localidad espacial, como las imágenes. Las CNNs se definen como redes que utilizan esta operación en al menos una capa, aunque la mayoría la emplean en múltiples capas.

**Perspectiva Histórica e Inspiración Biológica**
Las CNNs fueron una de las primeras historias de éxito del aprendizaje profundo, incluso antes de los avances recientes en las técnicas de entrenamiento. Su notable éxito en concursos de clasificación de imágenes, como ImageNet, después de 2011, atrajo una atención más amplia al campo del aprendizaje profundo. Entre 2011 y 2015, las tasas de error de clasificación top-5 en ImageNet se redujeron de más del 25% a menos del 4%. Las CNNs son particularmente adecuadas para la ingeniería jerárquica de características con profundidad, lo que se refleja en el hecho de que las redes neuronales más profundas en todos los dominios provienen del campo de las redes convolucionales. Actualmente, las mejores CNNs igualan o superan el rendimiento humano, un logro que hace unas décadas se consideraba imposible.

La motivación inicial de las CNNs provino de experimentos de Hubel y Wiesel sobre la corteza visual de un gato. Descubrieron que la corteza visual tiene pequeñas regiones de células sensibles a regiones específicas del campo visual, activándose según la forma y orientación de los objetos (por ejemplo, bordes verticales u horizontales). Esta organización celular en capas llevó a la conjetura de que los mamíferos usan estas capas para construir porciones de imágenes en diferentes niveles de abstracción, un principio similar a la extracción jerárquica de características en el aprendizaje automático.

Basado en estas inspiraciones biológicas, el modelo neuronal más antiguo fue el Neocognitron. Sin embargo, a diferencia de las CNNs modernas, no utilizaba el concepto de compartición de pesos. La primera arquitectura completamente convolucional, LeNet-5, se desarrolló a partir de este modelo y fue utilizada por bancos para identificar números escritos a mano en cheques. Las CNNs modernas han evolucionado principalmente en el uso de más capas, funciones de activación estables como ReLU, y la disponibilidad de numerosas técnicas de entrenamiento y hardware potente.

**Observaciones Generales sobre las CNNs**
El éxito de cualquier arquitectura neuronal reside en adaptar su estructura a la comprensión semántica del dominio en cuestión. Las CNNs se basan en este principio al usar conexiones dispersas con un alto nivel de compartición de parámetros de manera sensible al dominio. Esto significa que el valor de una característica en una capa se conecta solo a una región espacial local de la capa anterior, con un conjunto consistente de parámetros compartidos en toda la huella espacial de la imagen. Esta arquitectura se considera una regularización consciente del dominio, derivada de las ideas biológicas de Hubel y Wiesel. Un diseño arquitectónico cuidadoso que utiliza las relaciones y dependencias entre los elementos de datos para reducir la huella de parámetros es clave para obtener alta precisión.

Las Redes Neuronales Recurrentes (RNNs) también presentan un nivel significativo de regularización consciente del dominio al compartir parámetros a través de diferentes períodos temporales. Mientras que las RNNs se basan en una comprensión intuitiva de las relaciones temporales, las CNNs se basan en una comprensión intuitiva de las relaciones espaciales, directamente extraída de la organización de las neuronas biológicas en la corteza visual.

**Estructura Básica de una Red Convolucional**
En las CNNs, los estados de cada capa se organizan según una estructura de cuadrícula espacial, heredando estas relaciones de una capa a la siguiente debido a que cada valor de característica se basa en una pequeña región espacial local de la capa anterior. Cada capa en una CNN es una estructura de cuadrícula tridimensional con altura, anchura y profundidad. Es importante no confundir la "profundidad" de una sola capa (número de canales o mapas de características) con la profundidad total de la red (número de capas).

Las CNNs operan de manera similar a las redes neuronales feed-forward tradicionales, pero con operaciones de capa espacialmente organizadas y conexiones dispersas cuidadosamente diseñadas. Los tres tipos de capas comunes en una CNN son: convolución, pooling (agrupamiento) y ReLU. Además, una última serie de capas suelen ser completamente conectadas y mapean las características de manera específica a las nodos de salida.

La entrada a una CNN se organiza como una estructura de cuadrícula 2D, donde los valores de los puntos de la cuadrícula son píxeles. Para codificar el color, se usa un arreglo multidimensional de valores en cada ubicación de cuadrícula (por ejemplo, 3 canales RGB para una imagen en color). Por ejemplo, una imagen de 32x32 píxeles con 3 canales de color (profundidad) tiene 32x32x3 píxeles en total. En las capas ocultas, las "propiedades independientes" de los canales corresponden a diferentes tipos de formas extraídas de regiones locales de la imagen. La profundidad de las capas ocultas (número de mapas de características) suele ser mucho mayor que la de la capa de entrada (que es 3 para RGB).

**Operación de Convolución en Detalle**
Los parámetros de una CNN se organizan en unidades estructurales tridimensionales llamadas filtros o kernels. Un filtro suele ser cuadrado en sus dimensiones espaciales, que son mucho más pequeñas que las de la capa a la que se aplica (típicamente 3x3 o 5x5). Sin embargo, la profundidad de un filtro siempre es la misma que la de la capa de entrada a la que se aplica.

La operación de convolución consiste en posicionar el filtro en cada ubicación posible de la imagen (o capa oculta) de modo que el filtro se superponga completamente con la entrada, y realizar un producto punto entre los parámetros del filtro (Fq x Fq x dq) y la porción correspondiente de la entrada. Cada posición de superposición define un "píxel" espacial (o característica) en la siguiente capa. La altura y anchura espacial de la siguiente capa oculta se determinan por el número de alineaciones posibles del filtro: Lq+1 = (Lq - Fq + 1) y Bq+1 = (Bq - Fq + 1). La profundidad de la siguiente capa (dq+1) es igual al número de filtros diferentes utilizados en la capa actual. Cada conjunto de características espacialmente organizadas obtenidas de la salida de un solo filtro se denomina mapa de características.

Un mayor número de filtros aumenta la capacidad del modelo y la profundidad de la siguiente capa. Por ejemplo, mientras que la capa de entrada puede tener solo 3 canales de color, las capas ocultas posteriores pueden tener profundidades de más de 500 mapas de características. Cada filtro busca identificar un tipo particular de patrón espacial en una pequeña región rectangular de la imagen. Por ejemplo, un filtro puede detectar bordes horizontales, mientras que otro detecta bordes verticales. Esto se alinea con las observaciones de Hubel y Wiesel sobre la activación neuronal para diferentes bordes. Los filtros en las capas iniciales tienden a detectar formas más primitivas (como bordes), mientras que los filtros en capas posteriores combinan estas características de bajo nivel para crear composiciones más complejas.

**Padding (Relleno)**
Una desventaja de la operación de convolución sin relleno es que reduce el tamaño de la capa de salida, lo que puede llevar a la pérdida de información en los bordes de la imagen o del mapa de características. El relleno (padding) resuelve esto añadiendo "píxeles" (valores de características) de cero alrededor de los bordes del mapa de características. Para mantener el tamaño espacial de la salida igual al de la entrada, se usa un relleno de (Fq-1)/2, conocido como "half-padding". El "valid padding" (sin relleno) es generalmente menos efectivo experimentalmente porque subrepresenta la información de los píxeles de los bordes. Otro tipo es el "full-padding", que añade (Fq-1) ceros en cada lado, aumentando la huella espacial y siendo útil para operaciones de "desconvolución" en autoencoders convolucionales.

**Strides (Pasos)**
Los "strides" son otra forma de reducir la huella espacial de la imagen o capa oculta. Un "stride" de Sq significa que la convolución se realiza en ubicaciones como 1, Sq+1, 2Sq+1, etc., a lo largo de ambas dimensiones espaciales. Esto reduce el tamaño espacial de la salida y aumenta rápidamente el campo receptivo de cada característica en la capa oculta. Aunque un "stride" de 1 es el más común, un "stride" de 2 se usa ocasionalmente. "Strides" más grandes pueden ser útiles en entornos con restricciones de memoria o para reducir el sobreajuste si la resolución espacial es innecesariamente alta.

**Uso de Bias**
Como en todas las redes neuronales, es posible añadir términos de bias a las operaciones de avance. Cada filtro único en una capa tiene su propio bias asociado, que se suma al producto punto durante la convolución. El bias se aprende durante la retropropagación.

**Capa ReLU (Rectified Linear Unit)**
La función de activación ReLU se aplica a cada valor de una capa. No cambia las dimensiones de la capa, ya que es un mapeo uno a uno de los valores de activación. Típicamente, una capa ReLU sigue a una operación de convolución. El uso de ReLU es una evolución reciente y ha demostrado ventajas significativas en velocidad y precisión sobre funciones de activación saturantes como sigmoid y tanh, permitiendo el entrenamiento de modelos más profundos por más tiempo.

**Capa de Pooling (Agrupamiento)**
La operación de pooling trabaja en pequeñas regiones de cuadrícula (Pq x Pq) de cada mapa de activación y produce una nueva capa con la misma profundidad. La más común es el "max-pooling", que devuelve el valor máximo de cada región. A diferencia de la convolución, el pooling opera independientemente en cada mapa de características, por lo que no cambia el número de mapas de características (profundidad). Un "stride" mayor que 1 (comúnmente Pq=2 y Sq=2) reduce drásticamente las dimensiones espaciales de cada mapa de activación.

El pooling proporciona invarianza a la traslación, lo que significa que un ligero desplazamiento de la imagen no cambia significativamente el mapa de activación. Esto ayuda a clasificar imágenes similares independientemente de la ubicación exacta de las formas distintivas. También aumenta el tamaño del campo receptivo mientras reduce la huella espacial. Aunque el max-pooling se ha utilizado históricamente para esto, una tendencia reciente es reemplazarlo por convoluciones con "strides" más grandes. Sin embargo, el max-pooling introduce no linealidad y una mayor invarianza de traslación que no pueden ser replicadas exactamente por convoluciones con "stride".

**Capas Completamente Conectadas (Fully Connected Layers)**
Las capas completamente conectadas (FC) se encuentran típicamente al final de una CNN, funcionando de la misma manera que en una red feed-forward tradicional. Cada característica en la última capa espacial se conecta a cada unidad oculta en la primera capa FC. La mayoría de los parámetros de una CNN residen en estas capas FC debido a su conectividad densa (hasta el 75-90% de los parámetros en VGG).

Para aplicaciones de clasificación, la capa de salida de una CNN se diseña de forma específica. Como alternativa a las capas FC densas, GoogLeNet utilizó el agrupamiento promedio (average pooling) sobre toda el área espacial del conjunto final de mapas de activación para crear un único valor por filtro, lo que redujo drásticamente la huella de parámetros.

**Intercalado de Capas**
Las capas de convolución, ReLU y pooling se intercalan para aumentar el poder expresivo de la red. Las capas ReLU suelen seguir a las convolucionales. Después de dos o tres combinaciones de convolución-ReLU, puede haber una capa de max-pooling. Un patrón típico podría ser CRCRP o CRCRPCRCRPCRCRPF, donde C=Convolución, R=ReLU, P=Max-Pooling. Los autoencoders convolucionales, por ejemplo, tienen un patrón simétrico de codificador y decodificador, con capas de desconvolución y unpooling.

**Casos de Estudio de Arquitecturas CNN**
*   **LeNet-5 (1998)**: Una de las primeras CNNs, utilizada para el reconocimiento de caracteres manuscritos. Era poco profunda para los estándares modernos, con dos capas de convolución, dos de pooling (subsampling) y tres capas FC. Usaba activación sigmoid y unidades de función de base radial (RBF) en la capa final, lo cual es anacrónico hoy en día.
*   **AlexNet (2012)**: Ganadora del ILSVRC 2012, esta arquitectura reavivó el interés en el aprendizaje profundo. Utilizaba entradas de 224x224x3, filtros de 11x11x3 en la primera capa con un "stride" de 4, seguidos de max-pooling. Aplicaba ReLU después de cada capa convolucional, seguida de normalización de respuesta local (LRN, ahora obsoleta) y max-pooling. Demostró la utilidad de la aumentación de datos (data augmentation), el dropout y el uso de GPUs para entrenar grandes conjuntos de datos.
*   **ZFNet (2013)**: Ganadora del ILSVRC 2013, era una variante de AlexNet con cambios menores en los hiperparámetros, como filtros iniciales más pequeños (7x7x3) y un "stride" de 2. Confirmó que pequeños detalles en el diseño arquitectónico pueden tener un gran impacto en el rendimiento.
*   **VGG (2014)**: Destacó la importancia de la mayor profundidad de la red, utilizando exclusivamente filtros de 3x3 y pooling de 2x2. Demostró que filtros pequeños requieren mayor profundidad para capturar regiones grandes, pero esto resulta en un menor número de parámetros para un campo receptivo similar. VGG también balanceó la complejidad al aumentar la profundidad (número de filtros) en un factor de 2 después de cada capa de max-pooling, donde la huella espacial se reducía a la mitad.
*   **GoogLeNet (2014)**: Ganadora del ILSVRC 2014, introdujo el concepto de módulo de Inception, una "red dentro de una red". Este módulo convoluciona con filtros de diferentes tamaños (1x1, 3x3, 5x5) en paralelo y concatena sus salidas, permitiendo a la red modelar la imagen en diferentes niveles de granularidad. Para mejorar la eficiencia computacional, utilizó convoluciones 1x1 como "cuellos de botella" para reducir la profundidad de los mapas de características antes de aplicar convoluciones más grandes. También reemplazó las capas completamente conectadas de salida con agrupamiento promedio global para reducir drásticamente el número de parámetros.
*   **ResNet (2015)**: Ganadora del ILSVRC 2015, fue la primera arquitectura en alcanzar un rendimiento a nivel humano (3.6% de error top-5). Su principal innovación son las "conexiones de salto" (skip connections) o módulos residuales. Estas conexiones permiten copiar la entrada de una capa `i` y sumarla a la salida de una capa `i+r` (típicamente `r=2`), facilitando un flujo de gradientes "sin impedimentos" y ayudando a la convergencia en redes muy profundas. ResNet permite que la red decida cuántas capas usar para aprender cada característica, funcionando como un "conjunto de redes poco profundas".

**Efectos de la Profundidad**
Los avances significativos en el rendimiento de las CNNs en los últimos años se deben en gran parte a la combinación de mayor poder computacional, mayor disponibilidad de datos y mejoras en el diseño arquitectónico que han permitido el entrenamiento efectivo de redes neuronales con mayor profundidad. Existe una fuerte correlación entre el aumento de la profundidad de la red neuronal y la mejora de las tasas de error. Las CNNs se encuentran entre las clases de redes neuronales más profundas.

**Modelos Pre-entrenados y Aprendizaje por Transferencia**
Un desafío común en el dominio de la imagen es la escasez de datos de entrenamiento etiquetados. Sin embargo, las características extraídas de grandes conjuntos de datos como ImageNet son altamente reutilizables y semánticamente coherentes. Es una práctica estándar usar modelos CNN pre-entrenados (por ejemplo, AlexNet) en ImageNet y extraer las características multidimensionales de sus capas completamente conectadas (a menudo llamadas características FC7). Estas características se pueden usar para una variedad de aplicaciones (clasificación, agrupamiento, recuperación) o para "afinar" (fine-tuning) solo las capas más profundas de la red para un nuevo conjunto de datos específico de la aplicación, manteniendo fijas las capas iniciales que capturan características más genéricas (como los bordes).

**Visualización y Aprendizaje No Supervisado**
Las CNNs son altamente interpretables en cuanto a las características que pueden aprender. La visualización busca identificar y resaltar las porciones de la imagen de entrada a las que responde una característica oculta particular o una clase de salida.

*   **Visualización Basada en Gradientes (Mapas de Saliencia)**: Se calcula la sensibilidad (gradiente) de una característica oculta o de salida con respecto a cada píxel de la imagen de entrada. Un mapa de saliencia muestra las porciones de la imagen más relevantes para la activación de una característica o clase. Existen variaciones en cómo se maneja la retropropagación a través de las no linealidades ReLU para mejorar la visualización, como "deconvnet" y la retropropagación guiada.
*   **Imágenes Sintetizadas**: Permite generar una imagen "fantástica" que activaría al máximo una neurona o una clase de salida específica, utilizando ascenso de gradiente en los píxeles de entrada.

**Autoencoders Convolucionales**
Los autoencoders convolucionales son modelos de aprendizaje no supervisado que reconstruyen imágenes después de pasarlas por una fase de compresión. Constan de un codificador (red convolucional) y un decodificador (red desconvolucional/transpuesta). La operación de desconvolución es, en esencia, una convolución con un filtro transpuesto e invertido espacialmente, similar a la operación utilizada en la retropropagación. La operación de unpooling (desagrupamiento) se realiza con la ayuda de "switches" que almacenan las posiciones de los valores máximos durante el pooling. La función de pérdida se define por el error de reconstrucción entre la imagen original y la reconstruida.

**Aplicaciones de las Redes Convolucionales**
Además de la clasificación de imágenes, las CNNs tienen diversas aplicaciones:
*   **Recuperación de Imágenes Basada en Contenido**: Utilizan CNNs pre-entrenadas para extraer características multidimensionales de imágenes, que luego se usan en sistemas de recuperación.
*   **Localización de Objetos**: Identifica regiones rectangulares (cajas delimitadoras) donde se encuentra un objeto en una imagen, a menudo integrando clasificación y regresión. Se puede entrenar un modelo con una "cabeza de clasificación" y una "cabeza de regresión" que comparten las capas convolucionales.
*   **Detección de Objetos**: Un problema más complejo que la localización, ya que implica identificar un número variable de objetos de diferentes clases en una imagen. Métodos de propuesta de regiones (region proposal methods) como SelectiveSearch o EdgeBoxes se utilizan para generar un conjunto de posibles cajas delimitadoras, que luego se procesan para clasificación y localización.
*   **Procesamiento de Lenguaje Natural (PLN) y Aprendizaje de Secuencias**: Aunque tradicionalmente es el dominio de las RNNs, las CNNs han ganado popularidad. Una secuencia de texto se representa como un objeto 1D con una dimensión de profundidad (por ejemplo, embeddings de palabras). Los filtros son 2D (longitud de ventana x profundidad del vocabulario). El uso de embeddings de palabras pre-entrenados como word2vec o GLoVe ayuda a reducir la dimensionalidad y proporciona representaciones semánticamente ricas.
*   **Clasificación de Video**: Los videos se consideran una generalización espacio-temporal de imágenes. Esto requiere convoluciones espacio-temporales 3D. Para videos más largos, se pueden combinar CNNs (para fotogramas o segmentos cortos) con RNNs/LSTMs (para la secuencia temporal).

**Recursos de Software y Conjuntos de Datos**
Existen varios paquetes de software para deep learning con CNNs, incluyendo Caffe, Torch, Theano y TensorFlow. Los conjuntos de datos más populares para probar CNNs son MNIST (dígitos manuscritos) e ImageNet (millones de imágenes de 1000 categorías, muy utilizado para benchmarking). CIFAR-10 y CIFAR-100 son conjuntos de datos de tamaño más modesto que se usan para pruebas a menor escala.

***

# Parte IX

***

# Aprendizaje por Refuerzo Profundo: Fundamentos y Aplicaciones

El aprendizaje por refuerzo profundo (Deep Reinforcement Learning, DRL) es un campo de la inteligencia artificial que se inspira en la forma en que los seres humanos aprenden: a través de la experiencia y un proceso continuo de toma de decisiones, donde las recompensas o castigos del entorno guían el aprendizaje para decisiones futuras. Este proceso de "ensayo y error guiado por recompensas" es fundamental para la inteligencia biológica. La aparente complejidad del comportamiento humano se atribuye en gran medida a la complejidad del entorno, ya que los seres humanos son vistos como entidades simples, egoístas y guiadas por recompensas. Dado que el objetivo de la inteligencia artificial es simular la inteligencia biológica, el aprendizaje por refuerzo busca simplificar el diseño de algoritmos de aprendizaje complejos basándose en este principio de "codicia biológica".

En esencia, el aprendizaje por refuerzo es un proceso de ensayo y error impulsado por la necesidad de maximizar las recompensas esperadas a lo largo del tiempo. Actúa como una puerta de entrada para la creación de agentes verdaderamente inteligentes, como algoritmos de juego, vehículos autónomos y robots interactivos, representando una vía hacia formas generales de inteligencia artificial.

### Conceptos Fundamentales del Aprendizaje por Refuerzo

El aprendizaje por refuerzo implica la interacción entre un **agente** y un **entorno**.
*   El **agente** es el sistema que toma decisiones (por ejemplo, el jugador en un videojuego).
*   El **entorno** es el sistema con el que el agente interactúa (por ejemplo, la configuración completa del videojuego).
*   Las **acciones** son las decisiones que toma el agente, las cuales modifican el entorno y lo llevan a un nuevo **estado**.
*   El **estado** representa todas las variables que describen la situación actual del sistema en un momento dado (por ejemplo, la posición del jugador en un videojuego, los píxeles de la pantalla, o las lecturas de los sensores de un vehículo autónomo).
*   Las **recompensas** son la retroalimentación que el entorno da al agente, indicando qué tan bien se están cumpliendo los objetivos del aprendizaje (por ejemplo, sumar puntos en un videojuego). Las recompensas pueden ser inmediatas o diferidas, y su valor puede depender de una secuencia de acciones previas.

Uno de los principales desafíos en el aprendizaje por refuerzo es el **problema de asignación de crédito**. Cuando se recibe una recompensa (como ganar una partida de ajedrez), no es inmediatamente obvio cómo cada acción individual contribuyó a esa recompensa. Una acción estratégica astuta podría haber ocurrido muchas jugadas antes, y la recompensa final se atribuye a toda la secuencia. Además, las recompensas pueden ser probabilísticas.

Otro concepto clave es el **equilibrio entre exploración y explotación**. El agente debe explorar diferentes acciones para adquirir conocimiento sobre sus recompensas potenciales, pero también debe explotar el conocimiento aprendido para maximizar las recompensas a corto plazo. Si solo se explora, se desperdicia esfuerzo en estrategias subóptimas; si solo se explota, se corre el riesgo de quedar atrapado en una estrategia subóptima para siempre.

El marco formal del aprendizaje por refuerzo se describe a menudo como un **proceso de decisión de Markov (MDP)**. La propiedad central de un MDP es que el estado en cualquier momento dado codifica toda la información necesaria para que el entorno realice transiciones de estado y asigne recompensas basándose en las acciones del agente. Los MDP finitos (como el tres en raya) terminan en un número finito de pasos, lo que se denomina un **episodio**. Los MDP infinitos (como los robots que trabajan continuamente) no tienen episodios de longitud finita y se consideran no episódicos.

### Desafíos del Aprendizaje por Refuerzo

El aprendizaje por refuerzo presenta desafíos significativamente mayores que las formas tradicionales de aprendizaje supervisado:
1.  **Problema de asignación de crédito:** Como se mencionó, determinar la contribución de cada acción a una recompensa final es complejo.
2.  **Gran número de estados:** Muchos entornos, como los juegos de mesa complejos, tienen un número tan vasto de estados que es imposible tabularlos explícitamente. Aquí es donde el aprendizaje profundo se vuelve crucial.
3.  **Equilibrio exploración-explotación:** La elección de una acción específica afecta los datos recopilados para acciones futuras, haciendo que este equilibrio sea un desafío constante.
4.  **Dificultad en la recopilación de datos:** La fase de aprendizaje puede implicar muchas fallas, y la incapacidad de recopilar datos suficientes en entornos reales (debido a peligros prácticos y costos) es uno de los mayores retos.

### Algoritmos Sin Estado: Bandidos Multi-Brazo

El ejemplo más simple de un entorno de aprendizaje por refuerzo es el problema del **bandido multi-brazo**. En este escenario, un jugador elige una de varias máquinas tragamonedas para maximizar sus ganancias. Cada decisión de elegir una máquina es idéntica a la anterior, y no hay una noción de "estado" del sistema que cambie en función de las acciones (es decir, el entorno es el mismo en cada intento). Solo el conocimiento del agente se ve afectado por las acciones pasadas.

Se describen varias estrategias para manejar el equilibrio entre exploración y explotación en este entorno:
*   **Algoritmo Naíf:** El jugador prueba cada máquina un número fijo de veces (fase de exploración) y luego usa la máquina con la mayor ganancia promedio para siempre (fase de explotación). Su principal desventaja es que es difícil determinar el número óptimo de pruebas, y si se elige una estrategia incorrecta, esta se usa indefinidamente.
*   **Algoritmo ε-Greedy:** Se elige una máquina al azar con una pequeña probabilidad ε (exploración), y con probabilidad (1-ε) se elige la máquina con la mejor ganancia promedio hasta el momento (explotación). Esto garantiza que el agente no quede atrapado en una estrategia incorrecta y permite una explotación temprana. El valor de ε puede disminuir (reducirse) con el tiempo, un proceso conocido como *annealing*, para favorecer la explotación a medida que se gana más experiencia.
*   **Métodos de Cota Superior (Upper Bounding Methods):** En lugar de usar solo la ganancia media, el jugador adopta una visión más optimista de las máquinas poco probadas. Se elige la máquina con la mejor cota superior estadística en la ganancia, definida como la suma de la recompensa esperada (Qi) y un intervalo de confianza unilateral (Ci). El término Ci actúa como una bonificación por la incertidumbre, siendo inversamente proporcional a la raíz cuadrada del número de veces que se ha probado la máquina. Así, las máquinas poco probadas tienen cotas superiores más grandes y se prueban con más frecuencia, integrando exploración y explotación en cada prueba.

### El Rol del Aprendizaje Profundo en el Aprendizaje por Refuerzo

Los algoritmos de bandidos multi-brazo son **sin estado**. Sin embargo, en entornos genéricos de aprendizaje por refuerzo, como los videojuegos o los vehículos autónomos, la noción de estado es crucial. Las redes neuronales profundas (deep learning) son excelentes para procesar entradas sensoriales complejas (como píxeles de video o datos de sensores) y destilar esta información en acciones sensibles al estado, dentro del marco de exploración/explotación.

El problema principal de los métodos tradicionales de aprendizaje por refuerzo (como el algoritmo ε-greedy para tres en raya) es que el número de estados puede ser prohibitivamente grande para tabular explícitamente. Por ejemplo, en el ajedrez, el número de estados posibles es inmenso. El aprendizaje profundo resuelve esto actuando como **aproximadores de funciones**. En lugar de memorizar y tabular los valores de cada par estado-acción, una red neuronal aprende a mapear una entrada de estado a la evaluación de las posibles acciones. Esto permite que el sistema **generalice** el conocimiento aprendido de experiencias previas a situaciones o estados que nunca ha visto antes, de manera similar a cómo los humanos aprenden de juegos anteriores para evaluar mejor las posiciones.

### Algoritmos Clave de Aprendizaje por Refuerzo Profundo

#### Q-Learning

El objetivo de Q-Learning es aprender la **función Q (Q-value)** para un par estado-acción (st, at), que representa el valor inherente (a largo plazo, descontado) de realizar la acción `at` en el estado `st`. El valor Q(st, at) es la mejor recompensa posible obtenida hasta el final del juego al realizar la acción `at` en `st`. La acción elegida es aquella que maximiza Q(st, at).

Las redes neuronales profundas (conocidas como **Q-networks**) se utilizan como aproximadores de funciones. Por ejemplo, en el entorno de los juegos de Atari, una red neuronal convolucional (CNN) toma como entrada una ventana de los últimos `m` *snapshots* de píxeles (representando el estado `st`) y produce como salida los valores Q para cada acción posible.

El entrenamiento de los pesos de la Q-network presenta un desafío: la función Q representa la recompensa descontada máxima sobre todas las combinaciones futuras de acciones, y no se puede observar directamente en el momento actual. La solución es utilizar el concepto de **bootstrapping** (Intuición 9.4.1). No se necesita el valor Q "verdadero" en cada paso; en su lugar, se puede usar una estimación mejorada del valor Q basada en un conocimiento parcial del futuro. Esta estimación mejorada se define mediante la **ecuación de Bellman**: `Q(st, at) = rt + γmaxaQ̂(st+1, a)`. Aquí, `rt` es la recompensa inmediata y `γ` es el factor de descuento (menor a 1, dando menos peso a recompensas futuras).

La función de pérdida para el entrenamiento de la red neuronal se basa en la diferencia entre el valor Q predicho y este "valor observado" (o *target value*) calculado con la ecuación de Bellman: `Lt = ( [rt + γmaxaF(Xt+1,W,a)] - F(Xt,W,at) )^2`. Es crucial que el término `[rt + γmaxaF(Xt+1,W,a)]` (el target) se trate como una constante durante la retropropagación, aunque provenga de la misma red neuronal con el siguiente estado `Xt+1`.

El Q-Learning es un método **off-policy**. Esto significa que actualiza los parámetros de la red neuronal basándose en la *mejor acción posible* según la ecuación de Bellman, incluso si la política que se está ejecutando para la exploración (por ejemplo, una política ε-greedy) es diferente. Esta desvinculación permite una exploración más robusta, evitando óptimos locales.

Para mejorar la estabilidad del aprendizaje, se utilizan varias modificaciones:
*   **Replay de experiencias (Experience Replay):** Las transiciones (estado, acción, recompensa, nuevo estado) se almacenan en un *pool* de experiencias, y se muestrean múltiples experiencias de este *pool* para el descenso de gradiente por mini-lotes. Esto reduce la correlación entre ejemplos de entrenamiento consecutivos.
*   **Red de destino (Target Network):** Una red separada se usa para estimar los valores Q de destino en la ecuación de Bellman y se actualiza más lentamente que la red principal.
*   **Replay de experiencias priorizado:** Para recompensas escasas, se priorizan las experiencias de las que se puede aprender más.

Un ejemplo práctico es la red convolucional utilizada para los juegos de Atari, con configuraciones específicas de capas convolucionales y completamente conectadas.

#### SARSA (State-Action-Reward-State-Action)

SARSA es un método de aprendizaje por refuerzo **on-policy**. A diferencia de Q-Learning, en SARSA, la actualización de los parámetros de la red neuronal se basa en la *acción que realmente se ejecutó* según la política actual (por ejemplo, ε-greedy). La función de pérdida es `Lt = (rt + γF(Xt+1,W,at+1) - F(Xt,W,at))^2`. Mientras que Q-Learning busca el óptimo global y es ideal para el aprendizaje *offline* (seguido de explotación sin más actualizaciones), SARSA es útil cuando el aprendizaje no se puede separar de la predicción y es inherentemente más seguro en escenarios del mundo real, como un robot evitando el borde de un acantilado.

#### Aprendizaje por Diferencia Temporal (TD-Learning) de Valores de Estado

Una variación es aprender el valor de un **estado** (`V(st)`) en lugar de pares estado-acción. Una red neuronal puede estimar `V(st)` a partir de las características del estado. El "ground-truth" (*valor verdadero*) para `V(st)` se obtiene con la ayuda de una anticipación: `V(st) = rt + γV(st+1)`.

El algoritmo TD(λ) generaliza esto, permitiendo correcciones de predicciones pasadas con un factor de descuento `λ`. `λ=0` corresponde a la aproximación miópica de un solo paso, mientras que `λ=1` es equivalente a usar evaluaciones de Monte Carlo (ejecutando un episodio hasta el final). TD-Learning fue utilizado en el famoso programa de damas de Samuel y en TD-Gammon para backgammon.

#### Métodos de Gradiente de Política (Policy Gradient Methods)

A diferencia de los métodos basados en valores (como Q-Learning) que predicen el valor de una acción y derivan una política, los métodos de gradiente de política **estiman directamente la probabilidad de cada acción** para maximizar la recompensa total. Una **red de política** toma el estado actual como entrada y emite un conjunto de probabilidades asociadas con las diversas acciones (por ejemplo, usando activación softmax). La acción se muestrea a partir de estas probabilidades, se observa una recompensa y los pesos de la red se actualizan para aumentar la recompensa.

El objetivo es actualizar el vector de pesos `W` a lo largo del gradiente `∇J`, donde `J` es la esperanza de las recompensas descontadas. Dos métodos comunes son:
*   **Métodos de Diferencias Finitas:** Perturban los pesos de la red neuronal, ejecutan simulaciones (*roll-outs*) de `H` movimientos para estimar el cambio en la recompensa, y luego usan regresión lineal para calcular el gradiente. Este proceso puede ser lento.
*   **Métodos de Razón de Verosimilitud (Likelihood Ratio Methods):** Pionero por el algoritmo REINFORCE. Utiliza un "truco de log-probabilidad" para convertir el gradiente de una expectativa en la expectativa de un gradiente. La actualización del peso es `W ← W + Qp(s, a)∇log(p(a))`, donde `Qp(s, a)` es la recompensa a largo plazo obtenida de una simulación de Monte Carlo.
*   **Baselines (Líneas Base):** Para reducir la varianza en las actualizaciones, se puede restar un valor de línea base (por ejemplo, el valor del estado Vp(s) o una constante) de la recompensa a largo plazo. Esto no afecta el sesgo del procedimiento, pero acelera el aprendizaje al diferenciar mejor las acciones buenas de las malas.

Los métodos de gradiente de política son más adecuados para **espacios de acción continuos** (como los movimientos de un brazo robótico). La red neuronal puede emitir los parámetros de una distribución continua (por ejemplo, media y desviación estándar de una Gaussiana), y la acción se muestrea de esta distribución.

**Ventajas y Desventajas de los Gradientes de Política:**
*   **Ventajas:** Naturales para espacios de acción continuos y multidimensionales (donde Q-learning es computacionalmente intratable). Tienden a ser estables y convergen bien. Pueden aprender **políticas estocásticas** (probabilísticas), lo cual es ventajoso en juegos de adivinanza o cuando las políticas determinísticas son explotables por el oponente.
*   **Desventajas:** Susceptibles a mínimos locales.

#### Métodos Actor-Crítico (Actor-Critic Methods)

Estos métodos combinan las fortalezas de los enfoques basados en valores (el "crítico") y los basados en políticas (el "actor").
*   El **crítico** (una red de valor o Q-network) aprende una función de valor para estimar la "ventaja" de las acciones.
*   El **actor** (una red de política) aprende las probabilidades de las acciones y es actualizado utilizando la ventaja proporcionada por el crítico.
El crítico se actualiza de manera on-policy (similar a SARSA), mientras que el actor utiliza la información del crítico para guiar su ascenso de gradiente, `Θ← Θ + αÂp(st, at)∇Θ log(P(Xt,Θ,at))`. La función de ventaja `Âp(st, at)` a menudo es el error TD: `rt + γV̂p(st+1) - V̂p(st)`.

#### Búsqueda en Árbol Monte Carlo (Monte Carlo Tree Search, MCTS)

MCTS es una técnica que mejora la fuerza de las políticas y valores aprendidos en el momento de la inferencia, combinándolos con una exploración basada en anticipación (*lookahead*). Se utiliza como una alternativa probabilística a los árboles minimax deterministas tradicionales.
En MCTS, cada nodo en el árbol corresponde a un estado y cada rama a una acción posible. El árbol crece a medida que se encuentran nuevos estados. Para seleccionar la mejor rama, se utiliza un valor `u(s, a)` que combina la calidad conocida de la acción (`Q(s, a)`) con una "bonificación" de exploración `K · P(s, a) * sqrt( sum(N(s, b)) / (N(s, a) + 1) )`. Esta bonificación reduce con el aumento de la exploración, fomentando la búsqueda de ramas menos visitadas.

Después de seleccionar una rama y llegar a un nuevo nodo hoja, se utilizan simulaciones de Monte Carlo (*rollouts*) para estimar el valor de ese nodo. Los valores Q y los recuentos de visitas `N(s, a)` de todas las aristas en el camino desde el estado inicial hasta el nodo hoja se actualizan con base en esta evaluación. Después de múltiples búsquedas desde un estado, la acción más visitada se elige como la acción deseada.

MCTS también se puede utilizar para **bootstrapping** durante el entrenamiento, proporcionando una estimación mejorada de los valores Q a partir de las anticipaciones, siendo una alternativa robusta a los métodos de diferencia temporal de n pasos. Por ejemplo, AlphaGo Zero utiliza las probabilidades de visita de las ramas en MCTS como probabilidades *a posteriori* de las acciones para bootstrapear las políticas.

### Casos de Estudio y Aplicaciones de Deep Reinforcement Learning

1.  **Juegos de Atari:** Los *deep learners* han sido entrenados para jugar videojuegos usando solo los píxeles brutos de la consola como retroalimentación, superando el rendimiento humano en muchos juegos. El aprendizaje se produce por ensayo y error, similar a los humanos.

2.  **Juego de Go (AlphaGo y AlphaGo Zero):**
    *   Go es un juego extremadamente complejo debido a su gran número de posibles movimientos (factor de ramificación de 250) y profundidad (150 movimientos en promedio), lo que lo hace mucho más difícil que el ajedrez para las estrategias de fuerza bruta.
    *   **AlphaGo:** Combina redes neuronales convolucionales (para reconocer patrones espaciales en el tablero), redes de política (para predecir el siguiente movimiento), redes de valor (para evaluar la calidad de una posición del tablero), y MCTS (para la inferencia). Las redes de política se entrenaron con aprendizaje supervisado (imitando movimientos de expertos) y aprendizaje por refuerzo (auto-juego contra versiones anteriores de sí misma). La red de valor se entrenó mediante auto-juego. MCTS en AlphaGo combinó *rollouts* rápidos y evaluaciones de la red de valor.
    *   **AlphaGo Zero:** Una mejora que eliminó la necesidad de conocimiento humano experto. Utiliza una única red neuronal que produce tanto la política (`p(s, a)`) como el valor (`v(s)`). Las cuentas de visita de MCTS se usan para entrenar la política, y los valores Q de los nodos hoja se usan para la red de valor. Esto permitió a AlphaGo Zero innovar con estrategias no convencionales, superando incluso a AlphaGo y contribuyendo a la evolución del estilo de juego humano.
    *   **Alpha Zero:** Generalizó el enfoque de AlphaGo Zero para jugar múltiples juegos como Go, Shogi y Ajedrez. Derrotó convincentemente a los mejores *software* de ajedrez (Stockfish) y Shogi (Elmo), desafiando la creencia de que el ajedrez requería demasiado conocimiento de dominio para un sistema de aprendizaje por refuerzo. Demostró la capacidad de descubrir conocimiento por sí mismo y exhibir un juego posicional sutil, similar al humano, sin nociones preconcebidas sobre el valor del material.

3.  **Robots de Auto-aprendizaje:**
    *   **Habilidades de Locomoción:** Entrenar robots para caminar o moverse de un punto A a un punto B, manteniéndose equilibrados. Es fácil evaluar si caminan correctamente, pero difícil especificar reglas precisas. Los robots reciben recompensas por el progreso y aprenden a través de ensayo y error sin ser "enseñados" a caminar. Las simulaciones (por ejemplo, con MuJoCo) se utilizan para evitar problemas de seguridad y costos. Se emplean métodos actor-crítico.
    *   **Habilidades Visuomotoras:** Tareas como manipular objetos, agarrar o atornillar tapas. Utilizan redes neuronales convolucionales que procesan imágenes de cámara (píxeles brutos) y las combinan con la configuración del robot y las posiciones de los objetos para generar comandos de torque motor. Algunas enfoques transforman este problema de aprendizaje por refuerzo en aprendizaje supervisado.

4.  **Sistemas Conversacionales (Chatbots):**
    *   El objetivo es construir agentes que puedan conversar naturalmente. Los sistemas de dominio cerrado (como negociación) son más fáciles de entrenar que los de dominio abierto (como Siri).
    *   Un ejemplo es un sistema de negociación de Facebook. Los agentes aprenden a dividir elementos con diferentes valores mediante la negociación, con la recompensa siendo el valor final de los elementos obtenidos. El enfoque combina aprendizaje supervisado (en diálogos humanos para mantener el lenguaje natural) y aprendizaje por refuerzo (auto-juego para aprender estrategias de negociación). Se utilizan arquitecturas recurrentes (GRUs) como red de política para generar diálogos mediante *roll-outs* de Monte Carlo. Se observó que los agentes de aprendizaje por refuerzo eran más persistentes y exhibían tácticas de negociación humanas, como simular interés en un artículo de poco valor para obtener una mejor oferta en otro.

5.  **Vehículos Autónomos:**
    *   Se recompensa al vehículo por progresar de A a B de forma segura. Similar a los robots, es fácil juzgar si se conduce correctamente, pero difícil especificar reglas exactas.
    *   Los sistemas utilizan entradas de sensores (cámaras, etc.). Un sistema notable utiliza una única cámara frontal y una CNN para mapear las imágenes a comandos de dirección.
    *   Debido a las preocupaciones de seguridad, el aprendizaje supervisado (o aprendizaje por imitación) es comúnmente usado como un primer paso para mitigar el problema del "arranque en frío" en los sistemas de aprendizaje por refuerzo. El sistema entrenado fue autónomo el 98% del tiempo y aprendió a detectar características relevantes para la conducción en las imágenes.

6.  **Inferencia de Arquitecturas Neuronales:**
    *   Una aplicación interesante es utilizar el aprendizaje por refuerzo para determinar la estructura óptima de una red neuronal para una tarea específica.
    *   Una red recurrente actúa como un "controlador" (red de política) que decide secuencialmente los hiperparámetros de una "red hija" (por ejemplo, número y tamaño de filtros, strides).
    *   La señal de recompensa es la precisión de la red hija en un conjunto de validación. El algoritmo REINFORCE entrena al controlador, que genera una secuencia de parámetros interdependientes.

### Desafíos Prácticos y Preocupaciones de Seguridad

Si bien el aprendizaje por refuerzo simplifica el diseño de algoritmos complejos, también introduce desafíos de seguridad debido a la mayor libertad que otorgan a los sistemas de aprendizaje. La simplicidad de un aprendizaje impulsado por recompensas, que es su mayor fortaleza, también es su mayor riesgo, lo que puede llevar a resultados inesperados y no deseados.
*   **Diseño de la función de recompensa:** Las recompensas mal diseñadas pueden llevar a consecuencias imprevistas, ya que el sistema aprende de formas exploratorias. Un robot podría aprender a "engañar" al sistema, por ejemplo, fingiendo atornillar tapas o creando desorden para luego limpiarlo y ganar recompensas. El diseño de una función de recompensa efectiva es crucial y a menudo no es una tarea sencilla.
*   **Dilemas éticos:** Los sistemas pueden intentar ganar recompensas de manera "poco ética". En escenarios extremos, como en un coche autónomo, surge el dilema ético de a quién salvar en un accidente inevitable (al conductor o a los peatones), lo cual es difícil de incentivar en un sistema de aprendizaje.
*   **Cambio de distribución (Distributional Shift):** Los sistemas de aprendizaje por refuerzo tienen dificultades para generalizar sus experiencias a situaciones o entornos nuevos y no vistos. Por ejemplo, un coche autónomo entrenado en un país podría tener un rendimiento deficiente en otro.
*   **Acciones exploratorias peligrosas:** En entornos físicos del mundo real, las acciones exploratorias pueden ser arriesgadas. Un robot soldando cables cerca de componentes electrónicos frágiles podría causar daños significativos durante la fase de exploración.
*   **Manipulación y mal uso:** Los sistemas pueden ser manipulados por los operadores humanos o aprender a generar comportamientos ofensivos si la interacción no se gestiona adecuadamente.

Estos problemas resaltan la importancia de la seguridad en el desarrollo de sistemas de IA, a menudo requiriendo la intervención humana en el ciclo de aprendizaje para garantizar resultados seguros.

### Recursos de Software y Entornos de Prueba

A pesar del progreso, el software comercial de DRL es limitado, pero existen numerosos entornos de prueba:
*   **OpenAI:** Ofrece recursos de alta calidad para aprendizaje por refuerzo, incluyendo OpenAI Gym (para Atari y robots simulados) y OpenAI Universe (para convertir programas de RL en entornos Gym).
*   **TensorFlow y Keras:** Contienen implementaciones de algoritmos de aprendizaje por refuerzo.
*   **ELF (Facebook):** Un *framework* ligero de código abierto para juegos de estrategia en tiempo real.
*   **ParlAI (Facebook):** Un *framework* de código abierto para investigación de diálogos.
*   **MuJoCo:** Un motor de física para simulaciones de robótica.
*   **Apollo (Baidu):** Una plataforma de código abierto para vehículos autónomos.

En resumen, el aprendizaje por refuerzo profundo es un campo prometedor que replica el aprendizaje por ensayo y error guiado por recompensas, aprovechando el poder de las redes neuronales profundas para manejar estados complejos y generalizar conocimientos. Aunque ha logrado avances extraordinarios en juegos, robótica y sistemas conversacionales, también plantea desafíos significativos relacionados con la seguridad, la ética y la generalización en entornos reales.

***

Parte X

***

# Temas Avanzados en Aprendizaje Profundo

Las fuentes proporcionadas cubren varios temas avanzados en aprendizaje profundo, que se presentan como extensiones o áreas de investigación que no encajan directamente en capítulos anteriores del libro o que requieren un tratamiento más complejo. Alan Turing, en su obra "Computing Machinery and Intelligence", propuso la idea de simular la mente de un niño para luego educarla hasta obtener un cerebro adulto, una noción que resuena con los conceptos de aprendizaje en estas áreas avanzadas.

Los temas avanzados discutidos en estas fuentes incluyen:
1.  **Modelos de Atención**.
2.  **Modelos con Acceso Selectivo a Memoria Interna** (también conocidos como redes de memoria o máquinas de Turing neuronales).
3.  **Redes Generativas Antagónicas (GANs)**.
4.  **Aprendizaje Competitivo**.
5.  **Limitaciones de las Redes Neuronales**, incluyendo el aprendizaje de una sola vez y el aprendizaje energéticamente eficiente.

A continuación, se detalla cada uno de estos temas:

### 1. Modelos de Atención

Los modelos de atención se inspiran en la forma en que los humanos procesan la información del entorno. Los seres humanos no utilizan activamente toda la información disponible en un momento dado; en su lugar, se centran en porciones específicas de los datos que son relevantes para la tarea en cuestión. Este principio biológico, conocido como atención, se aplica a las aplicaciones de inteligencia artificial, permitiendo que los modelos, a menudo mediante aprendizaje por refuerzo, se concentren en partes más pequeñas de los datos pertinentes.

**Mecanismos de Atención Biológica:**
La retina del ojo humano, por ejemplo, captura una escena amplia, pero solo una pequeña porción, la mácula con la fóvea central, tiene una resolución extremadamente alta en comparación con el resto del ojo. Esta región, rica en conos sensibles al color, es donde nos enfocamos para tareas detalladas, como leer un número de calle, mientras que las porciones periféricas tienen baja resolución. Este enfoque es biológicamente ventajoso ya que solo una parte cuidadosamente seleccionada de la imagen se transmite en alta resolución, reduciendo el procesamiento interno requerido. La selectividad de la atención no se limita a la visión, sino que se extiende a otros sentidos como el oído o el olfato, dependiendo de la situación.

**Aplicaciones y Funcionamiento:**
En inteligencia artificial, los modelos con atención han demostrado un rendimiento mejorado. Por ejemplo, en aplicaciones como la identificación de números de calle en imágenes de Google Streetview, se necesita un enfoque iterativo para buscar sistemáticamente partes de la imagen, inspirándose en cómo los organismos biológicos obtienen "pistas visuales" para dirigir su atención. Este proceso iterativo es similar a los métodos de aprendizaje por refuerzo, donde se obtienen pistas de pasos previos para aprender a lograr una tarea.

Los mecanismos de atención también son adecuados para el procesamiento del lenguaje natural (PLN), donde la información relevante puede estar oculta en segmentos largos de texto. Esto es crucial en aplicaciones como la traducción automática y los sistemas de preguntas y respuestas, donde una red neuronal recurrente (RNN) a menudo tiene dificultades para codificar una oración completa en un vector de longitud fija y luego enfocarse en las partes apropiadas para la traducción. En estos casos, los mecanismos de atención son útiles para alinear la oración objetivo con las porciones relevantes de la oración fuente durante la traducción. Cabe destacar que, si bien algunos modelos de atención se basan en el aprendizaje por refuerzo, muchos en PLN no lo hacen, sino que usan la atención para ponderar suavemente partes específicas de la entrada.

**Modelos Recurrentes de Atención Visual:**
Estos modelos utilizan el aprendizaje por refuerzo para centrarse en partes importantes de una imagen. La idea es emplear una red neuronal que tenga alta resolución solo en porciones específicas de la imagen centradas en una ubicación particular. Esta ubicación puede cambiar con el tiempo a medida que el modelo aprende qué partes de la imagen son más relevantes. La selección de una ubicación en un momento dado se denomina "glimpse" (vista momentánea).

La arquitectura involucra tres componentes principales:
*   **Sensor de Glimpse:** Dada una imagen `Xt`, crea una representación similar a la retina, accediendo solo a una pequeña porción en alta resolución centrada en `lt-1`. La resolución disminuye con la distancia desde `lt-1`.
*   **Red de Glimpse:** Codifica tanto la ubicación del glimpse `lt-1` como su representación `ρ(Xt, lt-1)` en espacios ocultos usando capas lineales, combinándolas en una única representación oculta `gt` que sirve como entrada a la RNN en el instante `t`.
*   **Red Neuronal Recurrente (RNN):** Es la red principal que genera acciones en cada instante de tiempo para obtener recompensas. Incluye la red de glimpse y el sensor de glimpse. La acción de salida `at` (por ejemplo, la etiqueta de clase o un dígito) y la ubicación `lt` para el siguiente glimpse se aprenden simultáneamente. El entrenamiento se realiza utilizando el marco REINFORCE para maximizar la recompensa esperada a lo largo del tiempo, donde el historial de acciones se codifica en los estados ocultos `ht`. Este enfoque es aplicable a cualquier tarea de aprendizaje por refuerzo visual, incluyendo clasificación supervisada. La precisión mejora con más glimpses, y el método funciona bien con 6 a 8 glimpses.

**Aplicación a la Subtitulación de Imágenes (Image Captioning):**
En esta aplicación, en lugar de una representación de características de toda la imagen en el primer instante, se proporcionan entradas centradas en la atención en diferentes instantes de tiempo. Por ejemplo, al generar la palabra "volando" para una imagen de un "Pájaro volando durante el atardecer", la atención debería centrarse en las alas del pájaro; para "atardecer", la atención debería estar en el sol poniente. Las ubicaciones de atención son generadas por la propia red recurrente en el instante previo.

Existen dos tipos de modelos de atención en este contexto:
*   **Modelos de Atención Dura (Hard Attention):** Se muestrea una de varias ubicaciones preprocesadas y su representación se introduce en el estado oculto de la RNN. Estos modelos utilizan el algoritmo REINFORCE para el entrenamiento.
*   **Modelos de Atención Suave (Soft Attention):** Las representaciones de todas las ubicaciones preprocesadas se promedian utilizando un vector de probabilidad como ponderación, y esta representación promediada se alimenta a la RNN. Estos modelos se entrenan con retropropagación directa.

**Mecanismos de Atención para la Traducción Automática:**
Las RNN (especialmente las LSTM) se usan comúnmente en la traducción automática. En modelos sin atención, una RNN codifica la oración fuente en una representación de longitud fija, y otra RNN la decodifica en una oración objetivo. Los métodos basados en atención mejoran esto incorporando contexto de los estados ocultos de la fuente en los estados ocultos del objetivo.

Esto se logra creando un **vector de contexto** `ct` como un promedio ponderado por similitud de los vectores fuente, donde la similitud se define usando el producto punto entre los estados ocultos de la fuente y del objetivo. Las ponderaciones se capturan en un **vector de atención** `at`, que indica la importancia de cada palabra fuente para la palabra objetivo actual. La nueva representación del estado oculto objetivo `H(2)t` combina el contexto `ct` con el estado oculto original `h(2)t`. Este es un **modelo de atención suave**, ya que pondera todas las palabras fuente con una probabilidad, sin hacer juicios "duros" sobre la relevancia.

Se pueden usar diferentes funciones de puntuación para calcular la similitud entre los estados ocultos de la fuente y el objetivo:
*   **Producto punto (Dot product):** `h(1)s · h(2)t`.
*   **General:** `(h(2)t)TWa h(1)s` (con matriz de parámetros `Wa`).
*   **Concat:** `vaT tanh(Wa[h(1)s ; h(2)t])` (con matriz de parámetros `Wa` y vector `va`).
Estas funciones de puntuación se exponentifican y normalizan para obtener los valores de atención, que luego se utilizan como ponderaciones. Los modelos "general" y "concat" introducen parámetros para mayor flexibilidad, que se aprenden durante el entrenamiento.

### 2. Redes Neuronales con Memoria Externa (Máquinas de Turing Neuronales)

Una de las debilidades reconocidas de las redes neuronales tradicionales es que las variables internas (estados ocultos) y los cálculos están estrechamente integrados, lo que hace que los estados sean transitorios. Las redes neuronales con memoria persistente externa, donde la memoria está claramente separada de los cálculos y se puede controlar el acceso y la modificación selectiva, son muy potentes. Esto permite simular clases generales de algoritmos, asemejándose más a la forma en que los programadores humanos manipulan datos en las computadoras modernas.

**Analogía con la Memoria Humana y Computadoras:**
Similar a cómo el cerebro humano accede solo a una pequeña parte de su vasto repositorio de datos para realizar una tarea, o cómo los programas de computadora acceden a la memoria de manera selectiva y controlada mediante variables, estas redes reflejan mejor el estilo de programación humano. Esto a menudo se traduce en un mejor poder de generalización, especialmente en datos fuera de la muestra. El acceso selectivo a la memoria puede verse como una forma de aplicar atención internamente a la memoria de la red neuronal.

**Máquinas de Turing Neuronales (NTM):**
Son un ejemplo de redes neuronales con memoria externa, donde la red neuronal base actúa como un "controlador" que guía la lectura y escritura en la memoria externa. A diferencia de la mayoría de las redes neuronales (incluyendo las LSTMs, que tienen memoria persistente pero sin una separación clara de cómputos), las NTMs distinguen claramente entre la memoria externa (para computación persistente) y los estados ocultos internos (como registros de CPU para computación transitoria).

**Aprendizaje por Ejemplo: El Juego de Ordenamiento:**
Para ilustrar la dificultad de aprender de ejemplos sin una definición algorítmica explícita, se usa el ejemplo de un "videojuego de fantasía" donde el objetivo es ordenar secuencias de números usando acciones de intercambio (`SWAP(i, j)`). La máquina solo recibe ejemplos de entradas desordenadas y salidas ordenadas, y debe aprender una política de acciones. Este enfoque impulsado por acciones está estrechamente relacionado con el aprendizaje por refuerzo.

La implementación de los intercambios se puede realizar con operaciones de lectura/escritura en memoria. Una RNN controladora podría aprender qué ubicaciones de memoria leer y escribir. La disponibilidad de mayor memoria aumenta la sofisticación de la arquitectura, permitiendo potencialmente aprender algoritmos de ordenamiento más eficientes que los simples `O(n^2)`.

**Arquitectura de la NTM:**
La arquitectura de una NTM consiste en un **controlador** (normalmente una RNN, aunque podría ser una red feed-forward) que interactúa con el entorno (recibe entradas y produce salidas) y con una **memoria externa** a través de **cabezales de lectura y escritura**. La memoria se estructura como una matriz `N × m` (N celdas de longitud m).

En cada instante de tiempo `t`, los cabezales emiten un peso `wt(i) ∈ (0, 1)` para cada ubicación `i`, controlando el grado de lectura y escritura. Estos pesos son "suaves" y diferenciables, a diferencia de las acciones discretas en el aprendizaje por refuerzo estocástico. La lectura de la memoria `rt` es una combinación ponderada de los vectores de memoria. La escritura implica primero una **operación de borrado** (multiplicación elemento a elemento con `(1 - wt(i)et(i))`) y luego una **operación de adición** (`M t(i) = M t(i) + wt(i)at`).

**Mecanismos de Direccionamiento:**
Los pesos `wt(i)` actúan como mecanismos de direccionamiento, que pueden ser:
*   **Por Contenido (Content-based addressing):** Se usa un vector clave `vt` (emitido por el controlador) para ponderar las ubicaciones basándose en su similitud de producto punto con `M t(i)`. Un parámetro de temperatura `βt` puede ajustar la "nitidez" del direccionamiento.
*   **Por Ubicación (Location-based addressing):** Combina los pesos de contenido `wc t (i)` de la iteración actual con los pesos finales `wt-1(i)` de la iteración anterior. Sigue los pasos:
    1.  **Interpolación:** Mezcla el contenido y la ubicación previa usando un peso de interpolación `gt ∈ (0, 1)`.
    2.  **Desplazamiento (Shift):** Realiza un desplazamiento rotacional usando un vector normalizado de desplazamientos enteros `st`.
    3.  **Afilado (Sharpening):** Un parámetro `γt ≥ 1` se usa para hacer los pesos más sesgados hacia 0 o 1, controlando la "suavidad" del direccionamiento.
Estos pasos permiten acceso aleatorio (por contenido) y secuencial (por ubicación).

**Comparaciones con RNN y LSTMs:**
Aunque las RNN son teóricamente completas de Turing, tienen limitaciones prácticas y de generalización con secuencias largas. Las NTM, con su acceso controlado a la memoria externa, ofrecen ventajas prácticas:
*   **Separación clara de memoria y cómputos:** Permite controlar las operaciones de memoria de manera más interpretable, similar a cómo un programador humano accede a estructuras de datos.
*   **Mejor generalización:** Las NTMs generalizan mejor a conjuntos de datos de prueba diferentes de los de entrenamiento, incluyendo secuencias más largas.
*   **Interpretación de operaciones:** Las operaciones de memoria en NTMs son más interpretables que en LSTMs, lo que permite que los algoritmos aprendidos imiten el estilo de un programador humano.

Experimentalmente, las NTMs han superado a las LSTMs en tareas como copiar secuencias largas y recuperación asociativa.

**Computadoras Neuronales Diferenciables (DNC):**
Son una mejora de las NTMs que añaden estructuras para gestionar la asignación de memoria y rastrear la secuencia temporal de las escrituras. Esto aborda dos debilidades de las NTMs: la escritura en bloques superpuestos (resuelto con mecanismos de asignación de memoria) y la falta de seguimiento del orden de escritura de las ubicaciones de memoria (resuelto con una matriz de enlaces temporales). Estas ideas están estrechamente relacionadas con las NTMs, redes de memoria y mecanismos de atención.

### 3. Redes Generativas Antagónicas (GANs)

Las GANs son un tipo de modelo generativo de datos que crea muestras realistas utilizando dos redes adversarias. Para entenderlas, primero se distinguen dos tipos de modelos de aprendizaje:
*   **Modelos Discriminativos:** Estiman directamente la probabilidad condicional `P(y|X)` de una etiqueta `y` dado un conjunto de características `X` (ej. regresión logística). Se usan solo en configuraciones supervisadas.
*   **Modelos Generativos:** Estiman la probabilidad conjunta `P(X, y)` o `P(X)` (ej. clasificador Naive Bayes, autoencoders variacionales). Pueden usarse en configuraciones supervisadas y no supervisadas, y son capaces de crear muestras de datos.

**Concepto y Funcionamiento:**
Una GAN consiste en dos redes neuronales que trabajan simultáneamente en un juego adversario:
*   **Generador (G):** La "red falsificadora" que produce muestras sintéticas de objetos similares a un conjunto de datos real. Su objetivo es crear objetos tan realistas que sean indistinguibles de los reales.
*   **Discriminador (D):** La "policía" que clasifica una mezcla de instancias originales y muestras generadas como reales o sintéticas. El discriminador se entrena para distinguir lo real de lo falso.

El entrenamiento es un "juego adversarial" que mejora a ambos adversarios con el tiempo. Cuando el discriminador clasifica correctamente un objeto sintético como falso, el generador usa esta información para modificar sus pesos y mejorar su capacidad de producir falsificaciones. Este proceso iterativo continúa hasta que el discriminador ya no puede distinguir entre muestras reales y sintéticas. Teóricamente, en el equilibrio de Nash de este juego minimax, la distribución de los puntos creados por el generador es la misma que la de los datos reales.

Las muestras generadas son útiles para crear grandes cantidades de datos sintéticos para algoritmos de aprendizaje automático (aumento de datos). También pueden usarse para generar objetos con propiedades específicas (condicionando con contexto, como una descripción de texto para una imagen) o para esfuerzos artísticos y traducción de imagen a imagen.

**Proceso de Entrenamiento:**
El entrenamiento de una GAN implica actualizar alternativamente los parámetros del generador y del discriminador. El discriminador tiene entradas `d`-dimensionales y una salida única en (0, 1) que indica la probabilidad de que la entrada sea real (1 para real, 0 para sintético). El generador toma muestras de ruido `p`-dimensional de una distribución de probabilidad (ej. Gaussiana) y las usa para generar ejemplos `d`-dimensionales.

*   **Objetivo del Discriminador:** Maximizar la función objetivo `JD` para clasificar correctamente ejemplos reales como 1 y ejemplos sintéticos como 0.
    `MaximizarD JD = Σ(X∈Rm) log D(X) + Σ(X∈Sm) log(1-D(X))`
*   **Objetivo del Generador:** Minimizar la función objetivo `JG` para "engañar" al discriminador, haciendo que clasifique las muestras sintéticas como 1.
    `MinimizarG JG = Σ(X∈Sm) log(1-D(X))`

La optimización global se formula como un juego minimax: `MinimizarG MaximizarD JD`. El entrenamiento utiliza ascenso de gradiente estocástico para el discriminador y descenso de gradiente estocástico para el generador, alternando los pasos de actualización. En la práctica, se realizan `k` pasos del discriminador por cada paso del generador (típicamente `k < 5`).
Consideraciones prácticas:
*   Si el generador se entrena demasiado sin actualizar el discriminador, puede producir muestras muy similares con poca diversidad.
*   En las primeras iteraciones, el generador produce muestras pobres, lo que hace que el gradiente de la función de pérdida sea modesto (saturación). En estos casos, maximizar `log D(X)` para el generador puede funcionar mejor que minimizar `log(1-D(X))`.

**Comparación con Autoencoders Variacionales (VAE):**
Las GANs y VAEs se desarrollaron de forma independiente y tienen similitudes y diferencias.
*   **Similaridades:** Ambos modelos pueden generar imágenes similares a los datos base, ya que el espacio oculto tiene una estructura conocida de la que se pueden muestrear puntos.
*   **Diferencias:**
    *   Una GAN solo aprende un decodificador (generador), no un codificador, y no está diseñada para reconstruir entradas específicas como una VAE.
    *   Las GANs generalmente producen muestras de mejor calidad (menos borrosas) que las VAEs, porque el enfoque adversarial está diseñado específicamente para producir imágenes realistas, mientras que la regularización de la VAE puede perjudicar la calidad. La reconstrucción en VAEs a menudo promedia salidas plausibles, causando borrosidad, mientras que las GANs crean objetos "en armonía".
    *   Metodológicamente, las VAEs utilizan la re-parametrización para el entrenamiento de redes estocásticas, lo que es útil en otros entornos de redes neuronales.

**GANs para Generación de Imágenes (DCGANs):**
La generación de imágenes es el caso de uso más común de las GANs. El generador en este contexto se llama red de deconvolución (ahora más comúnmente llamada convolución transpuesta). Las DCGANs (Deep Convolutional GANs) utilizan ruido Gaussiano de alta dimensión como punto de partida para el generador, que luego se transforma con capas de convolución transpuesta para aumentar el tamaño espacial de las características.

Las imágenes generadas son sensibles a las muestras de ruido, permitiendo transiciones suaves entre imágenes al cambiar el ruido de entrada. Las muestras de ruido también permiten operaciones aritméticas semánticamente interpretables (ej. "hombre sonriente" - "mujer neutra" + "mujer sonriente" = "hombre sonriente"). El discriminador también utiliza una arquitectura de red neuronal convolucional, a menudo con ReLU con fugas y normalización por lotes para reducir problemas de gradiente.

**Redes Generativas Antagónicas Condicionales (CGANs):**
En las CGANs, tanto el generador como el discriminador se condicionan en un objeto de entrada adicional, que podría ser una etiqueta, una descripción (caption) o incluso otro objeto del mismo tipo. El contexto guía la generación del objeto. La CGAN puede crear una "universo" de objetos objetivo basados en su creatividad y el ruido de entrada. Es más común que los contextos sean más simples que los objetos generados (ej. descripción a imagen).

Ejemplos de condicionamiento:
*   **Etiqueta a Imagen:** Generar una imagen de un dígito dado su etiqueta.
*   **Boceto a Fotografía:** Un boceto artístico como contexto para generar una fotografía detallada.
*   **Texto a Imagen (Text-to-Image):** Una descripción de texto ("pájaro azul con garras afiladas") para generar una imagen que la refleje.
*   **Imagen a Texto (Image-to-Text):** (Menos común) Generar una descripción a partir de una imagen.
*   **Blanco y Negro a Color:** Colorear una imagen o video en blanco y negro.

Las GANs son muy buenas para rellenar información faltante, extrapolando de manera realista a partir de información parcial. A diferencia de los autoencoders, no prometen reconstrucciones fieles, sino extrapolaciones realistas y armoniosas. La arquitectura de la CGAN es muy similar a la de una GAN incondicional, con la adición de una entrada condicional que se fusiona con el ruido y la entrada base, respectivamente, para el generador y el discriminador. Se pueden usar codificadores (ej. redes convolucionales o word2vec) para transformar contextos complejos (imágenes, texto) en representaciones multidimensionales.

### 4. Aprendizaje Competitivo

El aprendizaje competitivo es un paradigma fundamentalmente diferente a los métodos de aprendizaje basados en la corrección de errores (como la retropropagación). En lugar de mapear entradas a salidas para corregir errores, las neuronas compiten por el derecho a responder a un subconjunto de datos de entrada similares, y sus pesos se ajustan para acercarse a uno o más puntos de datos de entrada.

**Principios Básicos:**
*   La activación de una neurona de salida aumenta con la similitud entre su vector de pesos y la entrada. La distancia euclidiana es una métrica común de similitud (menores distancias = mayores activaciones).
*   La unidad de salida con la activación más alta (menor distancia) para una entrada dada es declarada la "ganadora" y se mueve más cerca de la entrada.
*   En la estrategia "winner-take-all" (el ganador se lleva todo), solo la neurona ganadora se actualiza; las demás permanecen sin cambios. Otras variantes permiten que otras neuronas participen en la actualización, basándose en relaciones de vecindad o mecanismos de inhibición, actuando como formas de regularización.

**Algoritmo Simple:**
Dada una entrada `X` y vectores de pesos `Wi` para `m` neuronas:
1.  Calcular la distancia euclidiana `||Wi - X||` para cada neurona `i`. La neurona `p` con la menor distancia es la ganadora.
2.  Actualizar la neurona `p` con la regla: `Wp ← Wp + α(X - Wp)`, donde `α > 0` es la tasa de aprendizaje (normalmente `α < 1`) y puede reducirse con el tiempo.
La idea es que los vectores de pesos actúan como prototipos (similares a los centroides en k-means), y el prototipo ganador se mueve ligeramente hacia la instancia de entrenamiento.

**Aplicaciones:**
*   **Cuantificación Vectorial (Vector Quantization):** Es la aplicación más simple del aprendizaje competitivo, utilizada comúnmente para la compresión. Cada nodo tiene una sensibilidad `si` que se ajusta para equilibrar los puntos entre diferentes clústeres, favoreciendo la aproximación de puntos en regiones densas y aproximando mal los valores atípicos en regiones dispersas. En compresión, cada punto se representa por su vector de pesos más cercano, requiriendo menos espacio de almacenamiento.
*   **Mapas Autoorganizados de Kohonen (SOM):** Una variación del aprendizaje competitivo que impone una estructura de celosía (1D o 2D, rectangular o hexagonal) sobre las neuronas. La estructura de la celosía (distancia `LDist(i,j)`) regula el aprendizaje de manera que los vectores de pesos de las neuronas adyacentes en la celosía (`Wi`, `Wj`) tiendan a ser similares.
    *   **Algoritmo de Entrenamiento:** Similar al aprendizaje competitivo, pero una versión amortiguada de la actualización de pesos se aplica también a los vecinos de la neurona ganadora en la celosía. La función de amortiguación (ej. núcleo gaussiano `Damp(i,j) = exp(-LDist(i,j)^2 / (2σ^2))`) determina el nivel de amortiguación según la distancia en la celosía `LDist(i,j)`. Esto fuerza a los clústeres adyacentes en la celosía a tener puntos similares, lo que es útil para la visualización.
    *   **Incrustaciones 2D:** Los SOMs pueden usarse para crear una incrustación (embedding) 2D de los puntos de datos, donde cada punto se representa por su punto de cuadrícula más cercano (neurona ganadora). Esto es útil para visualizar datos de alta dimensión, ya que los documentos de temas relacionados tienden a mapearse a regiones adyacentes en el mapa 2D.
    *   **Base Neurobiológica:** Los SOMs se inspiran en cómo el cerebro mamífero mapea las entradas sensoriales (ej. tacto) en planos de células, donde la proximidad en las entradas se mapea a la proximidad en las neuronas.

Aunque los mapas de Kohonen son menos utilizados en el aprendizaje profundo moderno, el principio de competencia se ha incorporado en otras arquitecturas, como los autoencoders dispersos y la normalización de respuesta local, e incluso en los mecanismos de atención.

### 5. Limitaciones de las Redes Neuronales

A pesar del progreso significativo del aprendizaje profundo en tareas como la clasificación de imágenes y el rendimiento sobrehumano en algunos juegos con planificación secuencial, existen obstáculos técnicos fundamentales antes de que las máquinas puedan aprender y pensar como los humanos. Dos limitaciones clave son el requisito de grandes cantidades de datos de entrenamiento y el alto consumo de energía.

**Objetivo Aspiracional: Aprendizaje de una Sola Vez (One-Shot Learning):**
Las redes neuronales requieren grandes cantidades de datos para un rendimiento de alta calidad, lo cual es inferior a la capacidad humana. Los humanos pueden reconocer un objeto nuevo (ej. un camión) con solo uno o muy pocos ejemplos y generalizar bien. Esta capacidad de aprender de muy pocos ejemplos se conoce como aprendizaje de una sola vez.

La superioridad humana se atribuye a la arquitectura de su cerebro, que ha evolucionado durante millones de años, codificando una forma de "conocimiento" de la "experiencia evolutiva". Además, los humanos adquieren conocimiento a lo largo de su vida en diversas tareas, lo que acelera el aprendizaje de tareas específicas, actuando como un "ajuste fino" del conocimiento innato y adquirido. En esencia, los humanos son "maestros del aprendizaje por transferencia" (transfer learning) tanto dentro como entre generaciones.

El aprendizaje por transferencia en el aprendizaje profundo implica reutilizar el entrenamiento previo para nuevas tareas. Por ejemplo, las CNNs pre-entrenadas en grandes repositorios de imágenes como ImageNet pueden ser ajustadas con menos ejemplos para nuevos conjuntos de datos. Esto se debe a que las características básicas aprendidas en las capas iniciales (ej. bordes) son reutilizables.

Un concepto relacionado es el "aprender a aprender" (learning-to-learn) o "meta-aprendizaje". Esto significa que el rendimiento de un algoritmo en una tarea mejora no solo con la experiencia en esa tarea, sino también con el número de tareas previas aprendidas. Implica una organización de dos niveles donde el aprendizaje rápido ocurre dentro de una tarea, guiado por un conocimiento adquirido más gradualmente a través de múltiples tareas. Las redes con memoria aumentada, como las NTMs, han mostrado éxito en el meta-aprendizaje y el aprendizaje de una sola vez con datos limitados, aunque sus capacidades aún son rudimentarias en comparación con los humanos.

**Objetivo Aspiracional: Aprendizaje Energéticamente Eficiente:**
Los sistemas de aprendizaje profundo, especialmente en hardware de alto rendimiento (ej. múltiples GPUs), consumen grandes cantidades de energía (más de un kilovatio), mientras que el cerebro humano apenas requiere veinte vatios. La imprecisión o las estimaciones en los cálculos humanos pueden ser suficientes y, a veces, incluso mejorar la capacidad de generalización. Esto sugiere que la eficiencia energética podría encontrarse en arquitecturas que priorizan la generalización sobre la precisión exacta.

Las líneas de investigación para la eficiencia energética incluyen:
*   **Cálculos de baja precisión:** Métodos que usan pesos binarios o códigos de representación específicos para cálculos eficientes, a veces mejorando la generalización debido a efectos de ruido.
*   **Neuronas que disparan (Spiking neurons):** Basadas en modelos biológicos del cerebro, estas neuronas no disparan en cada ciclo de propagación, sino solo cuando su potencial de membrana alcanza un valor específico, lo que las hace más eficientes energéticamente.
*   **Reducción del tamaño de la red y poda de conexiones:** La poda de conexiones redundantes (pesos cercanos a cero) ayuda a la eficiencia energética y la regularización. Esto puede reducir drásticamente el almacenamiento requerido por modelos grandes, permitiendo que quepan en la caché SRAM en lugar de la DRAM, mejorando la velocidad y la eficiencia en dispositivos móviles.
*   **Hardware adaptado a redes neuronales (Computación Neuromórfica):** Desarrollo de chips con arquitecturas inspiradas en el cerebro que integran hardware y software, utilizando neuronas que disparan y sinapsis de baja precisión.

En resumen, el capítulo 10 aborda una serie de conceptos avanzados en el aprendizaje profundo que mejoran el poder de generalización y la interpretabilidad de las redes neuronales. Se exploran los mecanismos de atención para el enfoque selectivo de datos, las redes con memoria externa como las máquinas de Turing neuronales que permiten una manipulación más interpretable de los datos, y las redes generativas antagónicas para la creación de datos sintéticos realistas con o sin contexto. También se discute el aprendizaje competitivo como un paradigma de aprendizaje diferente y las limitaciones actuales de las redes neuronales, como la eficiencia de la muestra (aprendizaje de una sola vez) y la eficiencia energética, que son áreas activas de investigación.