# TP5 — Ejercicio 1.a: autoencoder básico

Estado del plan: confirmado por el usuario el 2026-11-08. La implementación comienza al invocar `assignment-step`; el presupuesto experimental se resolverá en el paso 3.

## Identidad y alcance

- Consigna: PA-004, `sources/practical-assignment/SIATP5.pdf`, p. 1, apartado 1.a; extracto `sources/extracted/PA-004.md` (rutas relativas al repositorio académico).
- Workspace: `tps_sia/tp5`, dentro del repositorio Git independiente `tps_sia`.
- Solo 1.a. Denoising (1.b) y VAE (2) quedan para planes posteriores.
- Fuente de datos: `data/font.h`, trasladada por pedido del usuario desde `C:\Users\Mateo\Downloads\font.h`, sin alterar su contenido. SHA256: `095B431BDFA2D228EEAA337FB45CD963088CC13380C189E01436E446767502F8`.
- La consigna exige representar 32 patrones de 5×7 en dos dimensiones, estudiar arquitecturas y optimización, graficar los códigos y generar una letra nueva. Solo permite recurrir a un subconjunto explicando por qué no se aprendieron todos los patrones.
- Nunca modificar originales académicos ni compilar contenido de esta solución en la wiki.

## Decisiones acordadas

1. Python, NumPy y Matplotlib; reutilizar el MLP y los optimizadores del TP3 mediante una interfaz en TP5.
2. Comandos Python, configuraciones JSON, modelos persistidos, métricas, gráficos y README reproducible.
3. Base: `35→20→2→20→35`, tanh en todas las capas internas (incluido el código), salida logística, error cuadrático y Adam. Es una hipótesis experimental, no garantía de convergencia.
4. Comparación acotada por etapas y varias semillas; el presupuesto y las variantes se acuerdan antes de ejecutar la comparación.
5. Reconstrucción binaria: salida `>= 0.5` es 1. Éxito: cada uno de los 32 patrones tiene como máximo un píxel incorrecto. Registrar también errores por patrón y error continuo.
6. Generación mediante interpolaciones y grilla latente. Comprobar novedad respecto de los 32 patrones y revisar visualmente con el usuario antes de afirmar que se generó una nueva letra.

No hay restricciones de librerías ni presupuesto de cómputo en la consigna. Las decisiones anteriores son acuerdos de implementación. No se crea un ADR: son elecciones locales reversibles.

## Fundamento y convenciones

- `wiki/topics/autoencoders.md`, secciones Funcionamiento y Conceptos centrales.
- OFF-022, `sources/official/Autoencoders.pdf`, extracto `sources/extracted/OFF-022.md`: pp. 3–6, arquitectura, reconstrucción y entrenamiento como MLP; pp. 28–38, exploración latente y sus límites.
- `../tp3/shared/mlp.py`: `MLP`, `predecir`, `entrenar`, `guardar`, `cargar`, `pesos`, `biases`, activaciones. La salida logística minimiza `0.5 * mean(sum((prediction-target)^2, axis=1))`; distinguir esta pérdida del MSE por píxel.
- `../tp3/shared/optimizers.py`: Adam y alternativas ya existentes.
- El entrenamiento del AE usa los 32 patrones tanto como entradas como objetivos. Es reconstrucción del conjunto solicitado; no presentar su error como una estimación de generalización a caracteres desconocidos.
- Nombres de archivos, módulos, claves y comandos nuevos en inglés; explicaciones del trabajo en español.
- Los comandos siguientes se ejecutarán desde `tps_sia/tp5`. Describen interfaces por construir, no comandos que ya estén disponibles.

## Pasos

### 1. Carga del dataset y estructura ejecutable

Estado: pending. Prerrequisito: confirmación de este plan. Entrada: `data/font.h`.

Objetivo: disponer de los 32 patrones originales sin transcripción manual ni alteración de la tipografía.

Procedimiento:

- Crear paquete `src/`, `tests/`, `configs/`, `requirements.txt` (NumPy y Matplotlib, compatible con TP3) y un README inicial con instalación.
- Implementar `src/dataset.py`: leer exclusivamente la matriz `Font3[32][7]`, ignorar comentarios al extraer sus números y validar 32 filas, 7 enteros por fila y valores entre 0 y 31. No ejecutar el header como código.
- Convertir cada entero en cinco bits, del bit 4 al bit 0 de izquierda a derecha; conservar el orden de filas de arriba abajo. Mantener imágenes `(32,7,5)` y matriz flotante `(32,35)` por filas. Identificar patrones por sus códigos `0x60` a `0x7f`; representar DEL con una etiqueta visible.
- Crear `src/plots.py` con una lámina del dataset y comando `python -m src.dataset --output results/dataset`.
- Probar forma, contenido binario, orden y correspondencia exacta entre enteros y bits; verificar casos asimétricos para detectar reflejos. Rechazar headers incompletos o mal formados.

Resultados: cargador, lámina original y pruebas de carga. Comprobación: `python -m unittest discover -s tests -p "test_dataset.py"`; comando de carga anterior. Aceptación: matriz `(32,35)`, lámina legible de 32 patrones, ninguna modificación de `font.h`.

Referencias: PA-004 p. 1, 1.a y 1.a.2; `data/font.h` completo.

### 2. Interfaz del autoencoder y evaluación

Estado: pending. Depende del paso 1.

Objetivo: reutilizar TP3 exponiendo reconstrucción, codificación y decodificación independientes.

Procedimiento:

- Implementar `src/autoencoder.py` y `src/metrics.py`. Resolver el import de `tp3.shared` desde la ubicación del archivo, con una única adaptación de ruta encapsulada; evitar importar un paquete ambiguo `shared`.
- Envolver un `MLP` con salida `logistica`, arquitectura base acordada y sin batch normalization, augmentation ni regularización adicional. Mantener todo el ajuste de pesos en el MLP existente; no duplicar backpropagation ni optimizadores.
- Exponer `encode(X)`, `decode(Z)`, `reconstruct(X)` y guardado/carga. Codificar aplicando pesos, biases y activación hasta la capa central; decodificar desde la siguiente capa hasta la salida logística usando los mismos parámetros. Validar que la capa central tenga exactamente dos unidades y que las dimensiones externas sean 35.
- Validar formas y finitud. No usar `predecir_clases` ni accuracy de clasificación para evaluar imágenes.
- Medir MSE por píxel, pérdida de entrenamiento con su normalización documentada, errores por patrón, máximo de errores, fracción de patrones exactos y fracción con como máximo un error.
- Probar `decode(encode(X)) == reconstruct(X)` antes y después de guardar/cargar; comprobar el caso exacto y casos con errores conocidos para las métricas, incluido el umbral 0.5.

Resultados: wrapper y métricas. Comprobación: `python -m unittest discover -s tests -p "test_autoencoder.py"` y `python -m unittest discover -s tests -p "test_metrics.py"`. Aceptación: código `(32,2)`, reconstrucción `(32,35)`, equivalencia numérica de ambas rutas y persistencia consistente.

Referencias: PA-004 p. 1, 1.a.1–2; OFF-022 pp. 3–6; wiki autoencoders, Funcionamiento; `../tp3/shared/mlp.py` y `../tp3/shared/optimizers.py`, interfaces citadas arriba.

### 3. Decisión del presupuesto y comparación

Estado: pending. Depende del paso 2. Es una puerta de decisión: preguntar y esperar antes de crear configuraciones ejecutables o entrenar.

Presentar al usuario opciones concretas con presupuesto por corrida y total, semillas, tasas, tamaños de lote, frecuencia de evaluación y reglas de parada. Considerar la pérdida del TP3 al proponer tasas. Opciones:

- Recomendada: piloto de la base con Adam para medir costo y evolución; luego comparar una red más ancha y otra más profunda manteniendo el cuello de botella de dos unidades; finalmente contrastar Adam con Momentum sobre la arquitectura seleccionada. Varias semillas compartidas entre candidatos.
- Alternativa: comparación factorial acotada de arquitecturas y optimizadores, con mayor costo y comparación más uniforme.
- Dejar abierto: pausa el entrenamiento y la comparación; no impide mantener las pruebas sintéticas de los pasos 1–2.

Registrar la elección y todos los valores en este plan antes de continuar. Si se necesita un piloto para estimar tiempos, acordar primero su presupuesto. Proponer selección por máximo error de píxeles, luego error medio de píxeles y finalmente MSE; informar todas las semillas y tasas de éxito, sin ocultar corridas fallidas. Acordar si se detiene al lograr el objetivo o se consume un presupuesto fijo para comparar.

Resultado y aceptación: protocolo acordado sin valores pendientes, y pasos 4–5 actualizados con los comandos y configuraciones exactos. Verificación: revisión del usuario. Sin pruebas de entrenamiento en este paso.

Referencias: PA-004 p. 1, 1.a.2; `../tp3/shared/optimizers.py`; acuerdo de comparación por etapas. Este paso resuelve decisiones diferidas expresamente por el usuario.

### 4. Entrenamiento base reproducible

Estado: pending. Depende del paso 3 y de su protocolo completo.

Objetivo: generar evidencia de una corrida inicial y disponer de un modelo reutilizable.

Procedimiento:

- Implementar `src/train.py`, configuración `configs/baseline.json` con los valores acordados y semilla explícita; incluir hash del dataset, arquitectura, configuración, versiones y duración en el manifiesto de ejecución.
- Entrenar con `X` como entrada y objetivo, manteniendo el estado de Adam entre evaluaciones. Guardar historia, modelo y métricas por patrón. Si se evalúa por bloques, comprobar que el entrenamiento continúa sin reiniciar pesos ni optimizador.
- No simular un conjunto de validación independiente con los mismos datos. Implementar las reglas de parada y selección acordadas usando métricas de reconstrucción; registrar claramente el motivo de parada.
- Producir curvas y lámina de originales/reconstrucciones con sus errores.
- Añadir prueba pequeña del flujo carga→entrenamiento→persistencia→evaluación; verifica consistencia sin exigir convergencia de una corrida corta.

Resultados: `results/baseline/` con modelo, manifiesto, historia, métricas y gráficos; instrucciones de reproducción. Comandos previstos: `python -m src.train --config configs/baseline.json --output results/baseline` y `python -m unittest discover -s tests`. Completar valores después del paso 3.

Aceptación: ejecución reproducible desde configuración, sin datos no finitos, modelo recargable y métricas recalculables. El objetivo académico se informa por separado: no considerar un fallo de convergencia como éxito ni usar el MSE como sustituto del límite por patrón.

Referencias: PA-004 p. 1, 1.a.2; OFF-022 pp. 3–6; `src/autoencoder.py`, `src/metrics.py`; `../tp3/shared/mlp.py`, `entrenar`, persistencia y estado; protocolo del paso 3.

### 5. Comparación experimental y revisión

Estado: pending. Depende del paso 4 y del protocolo del paso 3.

Objetivo: estudiar arquitecturas y optimización con evidencia, intentando reconstruir el conjunto completo.

Procedimiento: implementar `src/compare.py`, ejecutar exclusivamente el protocolo acordado, conservar resultados por configuración y semilla y generar tabla/gráficos con errores máximos, MSE, éxito, épocas y tiempo. Seleccionar el modelo según el criterio acordado; mantener trazabilidad hacia su corrida. Revisar resultados antes de ampliar la búsqueda. Si no se logra el objetivo, presentar evidencia y pedir la próxima decisión entre ampliar presupuesto, revisar arquitectura o justificar un subconjunto; no adoptar un subconjunto automáticamente ni afirmar imposibilidad matemática a partir de una búsqueda limitada.

Resultados: configuraciones y `results/comparison/` con resumen y referencia inequívoca al modelo elegido. Comando previsto: `python -m src.compare --config configs/comparison.json --output results/comparison` (configuración exacta tras el paso 3). Aceptación: corridas acordadas completas o interrupciones documentadas, comparación verificable y estado explícito del objetivo de 32 patrones.

Referencias: PA-004 p. 1, 1.a.2; protocolo del paso 3; artefactos del paso 4; OFF-022 pp. 3–6.

### 6. Espacio latente y candidatos nuevos

Estado: pending. Depende del paso 5 y de un modelo elegido.

Objetivo: representar los datos en dos dimensiones y explorar la generación.

Procedimiento: crear `src/explore.py`; graficar los 32 códigos con etiquetas, guardar coordenadas y generar interpolaciones entre códigos y una grilla en el rango ocupado. Con tanh, mantener las coordenadas dentro de `[-1,1]`. Hacer configurables resolución y pares a explorar, registrar coordenadas y modelo. Binarizar con el mismo umbral y calcular distancia de Hamming a cada patrón de entrenamiento. Marcar como copia toda muestra con distancia mínima cero; no llamar letra a cualquier imagen distinta. Presentar candidatos visuales al usuario y esperar su elección o revisión. Si no hay candidato adecuado, registrar el resultado y acordar otra exploración.

Resultados: `results/exploration/`, gráfico latente, grilla, interpolaciones y metadatos de novedad. Comando previsto: `python -m src.explore --config configs/exploration.json --output results/exploration`; esta configuración referencia explícitamente el modelo seleccionado del paso 5. Verificar que los códigos coincidan con `encode(X)` y que los metadatos reproduzcan las imágenes. La resolución visual es un parámetro de presentación; no modifica el modelo.

Aceptación: gráfico de 32 códigos, imágenes reproducibles y candidato diferente de todos los patrones; la afirmación de nueva letra requiere revisión visual del usuario. La revisión puede dejar este paso en progreso.

Referencias: PA-004 p. 1, 1.a.3–4; OFF-022 pp. 28–38; wiki autoencoders, Variantes y usos; modelo seleccionado del paso 5.

### 7. Documentación final y verificación

Estado: pending. Depende del paso 6, incluida la revisión visual.

Objetivo: dejar una solución explicable y reproducible del apartado 1.a.

Procedimiento: completar README y `results/report.md` con datos originales, arquitectura, normalización de pérdidas, protocolo, comparación, resultados por carácter, limitaciones y generación elegida. Citar PA para requisitos y OFF-022 para fundamentos, sin confundirlas. Registrar cualquier objetivo incumplido. Documentar instalación y todos los comandos efectivos desde TP5, incluidos imports del TP3. No redactar conclusiones experimentales antes de tener resultados.

Comprobación: `python -m unittest discover -s tests`; repetir evaluación y exploración desde el modelo guardado, sin repetir entrenamientos largos ya validados. Confirmar que cada obligación 1.a.1–4 tiene evidencia enlazada y que los originales permanecen intactos. Si se modificó código compartido del TP3, ejecutar además `python -m unittest discover -s shared/tests` desde `tps_sia/tp3`; evitar esas modificaciones cuando el wrapper local sea suficiente.

Resultado: documentación final y pruebas relevantes aprobadas, con límites explícitos. Referencias: PA-004 p. 1, 1.a completo; OFF-022 pp. 3–6 y 28–38; artefactos de los pasos 1–6.

## Continuación

Una vez confirmado este plan, invocar `assignment-step` para ejecutar exactamente el paso 1. No comenzar automáticamente los pasos siguientes. Actualizar estado y evidencia al terminar cada paso, conservando decisiones y trabajo realizado.
