# TP5 — Autoencoder básico para caracteres

## Problem Statement

El estudiante necesita resolver el apartado 1.a del TP5 de Sistemas de Inteligencia Artificial: aprender una representación de dos dimensiones de los 32 patrones binarios provistos por la cátedra, reconstruirlos dentro de la tolerancia requerida y explicar las arquitecturas y técnicas de optimización exploradas. También necesita visualizar esa representación y demostrar la generación de una nueva letra.

El espacio de trabajo del TP5 todavía no contiene una implementación. Ya está disponible el archivo original de caracteres y existe un MLP con optimizadores en el TP3 que se puede reutilizar. El estudiante necesita obtener resultados reproducibles y evidencia que permita explicar tanto los logros como las limitaciones del experimento.

La fuente de requisitos es PA-004, p. 1, apartado 1.a. El objetivo es aprender los 32 patrones de 5×7 con un máximo de un píxel incorrecto por patrón; la consigna admite un subconjunto únicamente si se explica por qué no fue posible aprender todo el conjunto.

## Solution

Construir un flujo de trabajo en Python y NumPy que permita cargar los caracteres originales, entrenar un autoencoder con código latente de dos dimensiones, evaluar sus reconstrucciones y comparar variantes por etapas. El estudiante ejecutará comandos con configuraciones JSON y obtendrá modelos persistidos, métricas, curvas y láminas de imágenes, junto con instrucciones para reproducirlos.

La solución comenzará con una arquitectura simétrica acordada y reutilizará el MLP y los optimizadores del TP3 mediante una interfaz propia del TP5. Después de revisar una comparación acotada con varias semillas, el estudiante podrá visualizar los códigos de los 32 patrones y explorar interpolaciones y una grilla latente. Se distinguirá un candidato nuevo de una nueva letra, cuya interpretación requiere revisión visual del estudiante.

La terminología sigue el [glosario del proyecto](../../../GLOSSARY.md). El [plan de ejecución confirmado](../../../tp5/assignment_plan.md) organiza la implementación en pasos; esta especificación describe el comportamiento y los resultados esperados.

## User Stories

1. Como estudiante, quiero cargar los caracteres del archivo original de la cátedra, para trabajar sobre el dataset solicitado sin transcribirlo manualmente.
2. Como estudiante, quiero conservar el orden y la orientación de los 32 patrones, para identificar correctamente cada carácter.
3. Como estudiante, quiero ver una lámina del dataset antes de entrenar, para detectar problemas de interpretación de los bits.
4. Como estudiante, quiero que una entrada incompleta o mal formada produzca un error comprensible, para evitar entrenar sobre datos incorrectos.
5. Como estudiante, quiero entrenar un autoencoder con una representación latente de dos dimensiones, para cumplir la restricción central del ejercicio.
6. Como estudiante, quiero comenzar con la arquitectura y las activaciones acordadas, para disponer de una base experimental concreta.
7. Como estudiante, quiero reutilizar el MLP y los optimizadores del TP3, para mantener continuidad con el trabajo que ya puedo explicar.
8. Como estudiante, quiero controlar las corridas mediante configuraciones JSON y semillas explícitas, para reproducir los experimentos.
9. Como estudiante, quiero acordar el presupuesto antes de ejecutar los experimentos, para controlar el tiempo destinado al entrenamiento.
10. Como estudiante, quiero observar la evolución del error durante el aprendizaje, para entender si una corrida mejora o se estanca.
11. Como estudiante, quiero guardar el modelo aprendido y volver a cargarlo, para analizarlo sin repetir el entrenamiento.
12. Como estudiante, quiero comparar visualmente cada original con su reconstrucción, para reconocer qué píxeles se reconstruyen incorrectamente.
13. Como estudiante, quiero conocer el número de errores de cada patrón, para identificar cuáles están reconstruidos dentro de tolerancia.
14. Como estudiante, quiero conocer el máximo error por patrón, para determinar si el conjunto está aprendido dentro de tolerancia.
15. Como estudiante, quiero registrar también el error continuo de reconstrucción, para explicar la evolución del entrenamiento sin confundirlo con el criterio de píxeles.
16. Como estudiante, quiero comparar arquitecturas de forma acotada y por etapas, para estudiar el efecto de su capacidad sobre la reconstrucción.
17. Como estudiante, quiero comparar técnicas de optimización con varias semillas, para distinguir resultados consistentes de una corrida afortunada.
18. Como estudiante, quiero conservar las configuraciones y los resultados de todas las corridas realizadas, para justificar la selección del modelo.
19. Como estudiante, quiero revisar los resultados antes de ampliar la búsqueda, para decidir el siguiente experimento con evidencia.
20. Como estudiante, quiero que los resultados incompletos queden documentados, para justificar las limitaciones sin presentar una búsqueda limitada como una prueba de imposibilidad.
21. Como estudiante, quiero ver los 32 códigos latentes con sus etiquetas en un gráfico de dos dimensiones, para relacionar la representación con los patrones originales.
22. Como estudiante, quiero explorar interpolaciones entre códigos, para observar cómo cambian las imágenes producidas por el decoder.
23. Como estudiante, quiero recorrer una grilla del espacio latente, para buscar candidatos nuevos y observar regiones poco estructuradas.
24. Como estudiante, quiero saber si una imagen generada coincide con algún patrón de entrenamiento, para distinguir una copia de un candidato nuevo.
25. Como estudiante, quiero revisar visualmente los candidatos nuevos, para decidir si alguno puede considerarse una nueva letra.
26. Como estudiante, quiero conservar las coordenadas y el modelo que produjeron un candidato, para regenerarlo posteriormente.
27. Como estudiante, quiero un README y un informe que relacionen requisitos, métodos y evidencia, para reproducir y presentar la solución del apartado 1.a.

## Acceptance Criteria

1. El dataset contiene exactamente los 32 patrones de la matriz original, cada uno con siete filas de cinco píxeles binarios. Se conserva el archivo original sin cambios y se respeta su orden.
2. Los caracteres conservan su orientación: los cinco bits de cada fila se leen del más significativo al menos significativo, de izquierda a derecha, y las filas se mantienen de arriba abajo. La lámina inicial permite revisar todos los patrones, incluidos los símbolos y DEL.
3. El cargador rechaza cantidades de patrones o filas incorrectas y valores que no caben en cinco bits, con un mensaje que permita identificar el problema.
4. La arquitectura base es `35→20→2→20→35`, con tanh interna, salida sigmoide, error cuadrático y Adam. Las variantes mantienen 35 entradas, 35 salidas y un código de dos dimensiones.
5. Las configuraciones y semillas efectivamente utilizadas quedan registradas junto con los resultados. El presupuesto, las variantes y las reglas de parada se acuerdan antes de ejecutar la campaña experimental.
6. El entrenamiento utiliza cada patrón como entrada y objetivo de reconstrucción. El informe distingue este ejercicio de una evaluación de generalización a caracteres desconocidos.
7. El código de los 32 patrones tiene forma `(32,2)` y las reconstrucciones continuas tienen forma `(32,35)`, con valores finitos. Decodificar los códigos reproduce la salida de reconstrucción del mismo modelo, dentro de tolerancia numérica.
8. Guardar y recargar el modelo permite reproducir sus códigos y reconstrucciones dentro de tolerancia numérica, sin entrenarlo nuevamente.
9. Cada salida continua se binariza con el criterio `valor >= 0.5`. Las métricas informan errores por patrón, máximo de errores, fracción de patrones exactos y fracción de patrones reconstruidos dentro de tolerancia.
10. El conjunto solo se declara aprendido dentro de tolerancia cuando los 32 patrones tienen individualmente como máximo un píxel incorrecto. Un promedio bajo o una pérdida continua pequeña no reemplazan este criterio.
11. Se informa el MSE por píxel y se documenta por separado la normalización de la pérdida empleada para entrenar. Las curvas y las láminas de reconstrucción son coherentes con las métricas guardadas.
12. La comparación incluye las arquitecturas, técnicas de optimización y semillas acordadas; conserva resultados de corridas fallidas o interrumpidas y permite identificar la corrida del modelo seleccionado.
13. Si no se alcanza el objetivo de los 32 patrones, el informe lo indica explícitamente. Adoptar un subconjunto o ampliar la exploración requiere revisar la evidencia y acordar la continuación; no se cambia el objetivo silenciosamente.
14. El gráfico latente contiene los códigos de los 32 patrones, identificables por sus etiquetas, y utiliza las mismas coordenadas que produce el modelo seleccionado.
15. Se producen interpolaciones y una grilla latente con imágenes y coordenadas reproducibles desde el modelo guardado.
16. Para cada candidato presentado se comprueba su igualdad o diferencia respecto de los 32 patrones binarios de entrenamiento. Solo una imagen distinta de todos ellos se denomina candidato nuevo.
17. La demostración de una nueva letra requiere un candidato nuevo y la revisión visual del estudiante. Si ningún candidato resulta adecuado, se registra esa limitación y la demostración queda pendiente.
18. El README describe instalación, configuración y comandos efectivos desde el espacio de trabajo del TP5. El informe enlaza la evidencia de los cuatro puntos del apartado 1.a y permite distinguir resultados obtenidos de objetivos pendientes.

## Implementation Decisions

- **Tecnología y reutilización:** Python, NumPy y Matplotlib. Una interfaz de autoencoder del TP5 encapsula el MLP y los optimizadores existentes del TP3; el entrenamiento y la retropropagación permanecen en esa implementación compartida.
- **Responsabilidades:** carga de datos, interfaz del modelo, métricas, ejecución del entrenamiento, comparación experimental y exploración visual. Estos límites organizan el trabajo sin exigir pruebas aisladas para cada componente interno.
- **Interfaz del modelo:** operaciones públicas de codificación, decodificación, reconstrucción, guardado y carga. Encoder y decoder utilizan los mismos parámetros que la reconstrucción completa; el cuello de botella consta de dos unidades.
- **Representación de datos:** cada patrón se conserva como imagen de siete filas por cinco columnas y se aplana por filas en un vector de 35 valores. La matriz de entrenamiento contiene 32 de esos vectores y sus etiquetas conservan los códigos originales.
- **Modelo inicial:** arquitectura `35→20→2→20→35`, tanh en todas las capas internas, incluida la latente, y salida logística. Sin batch normalization, aumento de datos ni regularización adicional en la base.
- **Pérdida:** conservar la salida logística y el error cuadrático del MLP del TP3. Su pérdida es la mitad de la suma de errores cuadrados por patrón, promediada entre patrones. El MSE por píxel se informa como otra magnitud con su propia normalización.
- **Entrenamiento y evaluación:** usar los 32 patrones como entradas y objetivos. La evaluación de reconstrucción usa errores de píxeles; las métricas de clasificación del MLP no determinan el cumplimiento del objetivo.
- **Configuración y ejecución:** comandos Python con JSON; registrar semilla, arquitectura, parámetros efectivos, identificación del dataset, versiones, duración y motivo de parada. Guardar modelo, historia, métricas y gráficos para poder revisarlos posteriormente.
- **Comparación por etapas:** comenzar por la base y continuar con una comparación acotada de arquitecturas y optimizadores con varias semillas. Los presupuestos, tasas, lotes, variantes concretas y criterios de selección se resolverán en la puerta de decisión del plan antes de ejecutar esa etapa; esta especificación no inventa sus valores.
- **Generación:** explorar interpolaciones y una grilla en el espacio latente del modelo elegido. El rango de tanh limita las coordenadas a `[-1,1]`. Conservar las coordenadas de los candidatos y comparar su versión binaria con todo el conjunto de entrenamiento.
- **Interpretación:** “patrón reconstruido dentro de tolerancia”, “conjunto aprendido dentro de tolerancia”, “candidato nuevo” y “nueva letra” tienen los significados del glosario. La novedad de una imagen no basta para considerarla una letra.
- **Convenciones:** archivos, módulos, claves y comandos nuevos en inglés; documentación explicativa en español. No modificar las fuentes académicas originales ni convertir resultados del TP en contenido de la wiki.
- **ADRs:** no hay un ADR aplicable identificado para esta funcionalidad. No se crea uno para estas decisiones locales reversibles.

## Testing Decisions

- El usuario confirmó como punto principal de prueba el flujo público del TP5 y sus artefactos: cargar datos, entrenar una corrida corta, guardar y recargar el modelo, reconstruir y explorar. Se priorizan resultados observables sobre la estructura interna de los módulos.
- Una prueba de integración pequeña verifica que ese flujo produce datos, códigos, reconstrucciones, métricas y archivos coherentes y recargables. No exige que una corrida corta aprenda los 32 patrones: ese resultado corresponde a la validación experimental con el presupuesto acordado.
- La carga se comprueba contra el contenido del dataset original y mediante entradas incompletas o mal formadas. Incluir patrones asimétricos permite detectar una inversión de bits o de orientación que una prueba de dimensiones no detectaría.
- Las comprobaciones puntuales de métricas usan reconstrucciones con errores conocidos para distinguir patrones exactos, patrones dentro de tolerancia y patrones fuera de tolerancia, incluido el valor de salida igual a 0.5.
- La interfaz pública del autoencoder se verifica mediante la equivalencia entre reconstrucción y codificación seguida de decodificación, antes y después de persistir el modelo. No se comprueban listas privadas de activaciones ni detalles del algoritmo de retropropagación desde el TP5.
- La exploración se comprueba regenerando las imágenes a partir del modelo y las coordenadas guardadas y verificando la novedad frente al dataset completo. La identificación como nueva letra se valida visualmente con el estudiante.
- La evidencia experimental verifica el máximo de errores por patrón sobre los 32 caracteres y conserva los resultados por semilla. No se convierte la convergencia de una semilla arbitraria en una condición frágil de las pruebas automáticas.
- Como antecedentes, el TP3 ya contiene pruebas de gradientes y comportamiento del MLP, optimizadores, carga de dígitos, persistencia y continuidad del entrenamiento. Se aprovecha esa cobertura en lugar de duplicar la comprobación de sus algoritmos en TP5. Si se modifica código compartido, corresponde ejecutar sus verificaciones existentes mediante sus puntos de entrada efectivos.
- Las pruebas nuevas cubren carga, modelo, métricas y ejecución por su comportamiento externo. No se agregan interfaces de prueba adicionales por cada detalle interno.

## Out of Scope

- Denoising autoencoder del apartado 1.b y estudios de niveles de ruido.
- VAE, nuevos datasets y generación variacional del ejercicio 2.
- Migración a PyTorch u otro framework de entrenamiento.
- Reescritura del MLP, sus optimizadores o una reorganización general del TP3.
- Una interfaz web o un notebook como punto principal de ejecución.
- Una búsqueda amplia de hiperparámetros sin revisión previa del presupuesto.
- Declarar generalización a letras desconocidas a partir de la reconstrucción del conjunto de entrenamiento.
- Modificar la wiki o los originales de las fuentes del curso.

## Further Notes

- Especificación sintetizada de las decisiones confirmadas en la conversación. El enfoque de pruebas también fue confirmado expresamente por el usuario.
- El archivo original de caracteres ya fue trasladado al espacio de trabajo del TP5 por instrucción del usuario. Su hash registrado en el plan permite comprobar que se conserva intacto.
- La arquitectura base es una hipótesis de partida, no un resultado experimental. Todavía no hay evidencia de convergencia ni una nueva letra validada.
- El presupuesto de entrenamiento y el protocolo concreto de comparación permanecen diferidos explícitamente a la puerta de decisión del plan. La estructura de datos y la interfaz del modelo pueden implementarse antes de resolverlos.
- La consigna se identifica como **PA-004**, aunque el número del trabajo práctico es **TP5**. Fuente de requisitos: [consigna original, p. 1](../../../../sources/practical-assignment/SIATP5.pdf) y [extracto](../../../../sources/extracted/PA-004.md).
- Fundamento docente: [wiki de autoencoders](../../../../wiki/topics/autoencoders.md), secciones Conceptos centrales y Funcionamiento; [OFF-022, extracto](../../../../sources/extracted/OFF-022.md), pp. 3–6 sobre arquitectura y reconstrucción y pp. 28–38 sobre exploración latente y sus límites. La consigna define las tareas; las fuentes docentes sustentan las explicaciones.
