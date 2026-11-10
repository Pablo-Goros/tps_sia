# TP5 — Planes de autoencoders

El plan vigente de denoising está en [Ejercicio 1.b](#ejercicio-1b--denoising-autoencoder).
La sección de 1.a se conserva como registro previo: sus estados y rutas históricos
no describen necesariamente el estado actual de este worktree.

## Ejercicio 1.a: autoencoder básico (plan previo)

## Extensión autorizada: rho de RMSProp (2026-11-10)

Estado: completed. El usuario aprobó la propuesta de comparar rho=0.85 y 0.95
a tasa fija 0.002, semillas 42/43/44 desde cero, frente a las tres referencias
originales de rho=0.9. Máximo seis corridas nuevas y 18000 épocas; no autoriza
experimentos posteriores ni reemplazar la selección final.

Objetivo: comprobar si cambiar la memoria del promedio móvil de gradientes
cuadráticos mejora la proporción de semillas que cumplen. Mantener arquitectura
deeper, tanh en código bidimensional, salida logística, umbral 0.5, lote 32,
epsilon=1e-8 y ausencia de L2, augmentation y batch normalization.

Entradas: `configs/comparison-rmsprop-rho.json`, `data/font.h`, modelos originales
`results/comparison-broad/deeper-rmsprop-0p002-seed-{42,43,44}/`.
Referencias: PA-004 p. 1, 1.a.2; OFF-013 pp. 8–10; wiki autoencoders,
Funcionamiento; `tp3/shared/optimizers.py`, RMSProp (ruta desde `tps_sia/`);
`src/compare.py`, `src/train.py` y `comparison.md`.

Procedimiento:

1. Validar protocolo con `src.compare._protocol` y construcción de optimizadores;
   verificar seis corridas nuevas, tres reutilizadas y cap de 18000 épocas.
   Comprobar hashes, dataset y métricas recargadas de las referencias. Registrar
   hashes de los archivos de `comparison-broad/`, `final/`, la continuación y
   la comparación local de tasas para comprobar su conservación al terminar.
2. Usar Python Windows 3.13.7, NumPy 2.5.1 y Matplotlib 3.11.2, con variables
   BLAS/OMP sin fijar como en los manifiestos originales. No comparar velocidad.
3. Desde `tp5/`, ejecutar una sola vez
   `python -m src.compare --protocol configs/comparison-rmsprop-rho.json --output results/comparison-rmsprop-rho-2026-11-10`.
   Cada corrida nueva tiene máximo 3000 épocas, evaluación cada 100 y parada
   al cumplir los 32 patrones. Reutilizar las tres referencias de rho=0.9 a
   su presupuesto original, sin entrenarlas ni extenderlas. Sin reintentos.
4. Recargar los nueve modelos y comprobar métricas, predicciones, códigos,
   historias, actualizaciones, épocas, motivos de parada y configuraciones
   efectivas: solo debe variar rho. Verificar presupuesto y hashes originales.
5. Documentar resultados por corrida y agregados por rho: éxitos (objetivo
   exploratorio: superar 1/3), máximo error medio, error medio y MSE. Mantener
   también el criterio agregado histórico y explicar cualquier discrepancia.
   No extrapolar tres semillas ya usadas a robustez con semillas nuevas.
   Conservar `final/selection.json` y detener el trabajo experimental.

Salidas: protocolo, comparación de nueve filas con seis entrenamientos nuevos,
modelos/historias/métricas nuevos, verificaciones y documentación en
`comparison.md`. Aceptación: presupuesto respetado, seis corridas completas o
interrupción documentada, métricas reproducibles al recargar y archivos previos
intactos. No requiere cambios de código ni nuevas pruebas que dupliquen JSON.

Resultado: seis corridas nuevas completas, 18000 de 18000 épocas máximas,
412.22 s sumando las duraciones de esas corridas. Ambos rho nuevos obtuvieron
0/3 éxitos y máximo error 2 en todas las semillas. Rho=0.85 redujo error medio
y MSE agregados, pero no resolvió los patrones fuera de tolerancia. La referencia
rho=0.9 mantiene 1/3 y gana en máximo error medio (1.6667 frente a 2.0).
Se verificaron por recarga nueve modelos, variación exclusiva de rho en sus
configuraciones efectivas y 2123 archivos anteriores intactos por SHA256.
`final/` conserva la semilla 44 original. No se ejecutaron otros experimentos.

## Extensión autorizada: tasas RMSProp cercanas (2026-11-10)

Estado: completed. El usuario pidió continuar con la siguiente propuesta ya
presentada: tasas fijas 0.0015 y 0.0025 desde cero, tres semillas por tasa,
máximo 18000 épocas nuevas. Este paso no autoriza otros experimentos.

Objetivo e hipótesis: comprobar si una tasa cercana a 0.002 mejora la trayectoria
completa y la proporción de semillas que cumplen, frente a la continuación tardía
que no resolvió los patrones pendientes. Mantener arquitectura deeper, tanh en
código de dos dimensiones, salida logística, umbral 0.5, lote 32, rho=0.9,
epsilon=1e-8 y ausencia de L2, augmentation y batch normalization.

Entradas: `configs/comparison-rmsprop-local.json`, `data/font.h`, modelos y
configuraciones `results/comparison-broad/deeper-rmsprop-0p002-seed-{42,43,44}/`.
Referencias: PA-004 p. 1, 1.a.2; OFF-013 pp. 7–10; wiki autoencoders,
Funcionamiento; `comparison.md`; contratos de `src/compare.py` y `src/train.py`.

Procedimiento:

1. Validar el protocolo con `src.compare._protocol`, verificar el modelo/dataset
   y métricas recargadas de las tres referencias y registrar hashes de todos los
   archivos de `comparison-broad/`, `final/` y la continuación anterior.
   Comprobar seis corridas nuevas, tres reutilizadas y máximo 18000 épocas nuevas.
2. Usar el Python Windows 3.13.7, NumPy 2.5.1 y Matplotlib 3.11.2 originales.
   Mantener las variables de hilos BLAS/OMP sin fijar, como constan efectivamente
   en los manifiestos originales. No extraer conclusiones de las duraciones.
3. Desde `tp5/`, ejecutar una sola vez
   `python -m src.compare --protocol configs/comparison-rmsprop-local.json --output results/comparison-rmsprop-local-2026-11-10`.
   Cada tasa nueva usa semillas 42/43/44, máximo 3000 épocas por corrida,
   evaluación cada 100 y parada al cumplir la tolerancia. Reutilizar las tres
   corridas originales de 0.002 a su presupuesto original, sin continuar ni
   repetirlas. Sin reintentos automáticos.
4. Recargar los seis modelos nuevos y comprobar métricas, historias, épocas,
   actualizaciones, tasas y configuraciones efectivas. Verificar presupuesto y
   hashes de todos los archivos anteriores.
5. Documentar tabla por corrida y agregados por tasa: proporción de éxitos,
   máximo error medio, error medio, MSE y épocas. Mantener el criterio histórico
   de selección por medias, describiendo cualquier divergencia con la proporción
   de éxitos. Tres semillas ya usadas sirven para exploración; no demuestran
   robustez sobre semillas nuevas. Mantener intacta `final/selection.json`.

Salidas: protocolo, nueve filas de comparación (seis entrenadas y tres
reutilizadas), modelos/historias nuevos, resumen, comprobación de recarga y
preservación, informe en `comparison.md`. Aceptación: seis corridas completas o
interrupción documentada, no superar 18000 épocas nuevas, métricas reproducibles
al recargar y originales intactos. No requiere modificar código ni agregar
pruebas que repliquen el esquema JSON; se valida contra el comparador existente.

Resultado: seis corridas nuevas completas, 17500 de 18000 épocas máximas,
411.64 s sumando las duraciones de las seis corridas. La tasa 0.0015 obtuvo
0/3 éxitos; 0.0025 obtuvo 1/3, con semilla 44 exitosa a época 2500. La referencia
0.002 reutilizada mantiene 1/3 y gana según el orden agregado histórico:
máximo error medio 1.6667, error medio 0.45833 y MSE medio 0.01328958. No hubo
mejora de consistencia entre estas semillas. Se verificaron por recarga los
nueve modelos y 2041 archivos anteriores intactos por SHA256. `final/` conserva
su selección original. No se ejecutaron experimentos posteriores.

## Extensión autorizada: continuación RMSProp (2026-11-10)

Estado: completed. Este paso independiente sigue al experimento factorial ya
completado; no altera los estados históricos de los pasos originales ni autoriza
campañas posteriores. El usuario aprobó el protocolo propuesto en conversación.

Objetivo: comparar, para las semillas 42 y 43 pendientes, continuar a tasa 0.002
contra reducirla a 0.001 desde sus modelos a época 3000. La semilla 44 ya cumple
y no se vuelve a entrenar. Hipótesis: una reducción tardía puede corregir los
últimos píxeles; el control distingue ese efecto del presupuesto adicional.

Entradas: `results/comparison-broad/deeper-rmsprop-0p002-seed-{42,43}/`,
`results/final/selection.json`, `configs/continuation-rmsprop.json`, `data/font.h`.
Fundamentos: PA-004 p. 1, 1.a.2; wiki autoencoders, Funcionamiento;
OFF-013 pp. 7–10, ajuste de tasa y RMSProp; `comparison.md`.

Procedimiento:

1. Exponer cambio de tasa conservando el optimizador y un comando separado de
   continuación. Verificar hash, dataset, métricas originales, momentos,
   contador, RNG e historia antes de actualizar parámetros.
2. Comprobar equivalencia entre continuación en memoria y tras recarga, tasa
   reducida, conservación del origen, rechazo de sobrescritura y presupuesto
   vencido con `python -m unittest discover -s tests -p "test_continuation.py"`.
   Ejecutar también `python -m unittest discover -s tests` desde `tp5/`.
3. Desde `tp5/`, con OPENBLAS_NUM_THREADS=1, ejecutar
   `python -m src.continue_training --protocol configs/continuation-rmsprop.json --output results/continuation-rmsprop-2026-11-10`.
   Usar Python 3.13.7 / NumPy 2.5.1 / Matplotlib 3.11.2 disponibles en Windows,
   iguales a las versiones registradas en la campaña original.
4. Ejecutar dos semillas por dos tasas, hasta 1000 épocas adicionales por rama:
   máximo 4000 épocas adicionales y 600 segundos compartidos de ejecución,
   comprobando tiempo antes de cada época. Evaluar cada 100 épocas y detener
   cada rama al cumplir todos los patrones. Sin reintentos automáticos. El
   cierre y guardado de evidencia se completan aun si vence el tiempo.
5. Comparar éxito por semilla y tasa, épocas hasta éxito y errores por patrón;
   informar también medias de máximo error, error medio y MSE según el criterio
   anterior. No atribuir una mejora exclusiva de MSE a cumplimiento binario.
6. Documentar resultados, comprobar hashes de los archivos originales y detener
   el trabajo experimental. Mantener `final/` y su selección vigente intactos.

Salidas: código y pruebas de continuación, protocolo, preflight, cuatro ramas
con modelo/historia/métricas/procedencia y resumen en directorio nuevo; informe
en `comparison.md`. Aceptación: presupuesto respetado o interrupción documentada,
resultados reproducibles al recargar, originales intactos y conclusión limitada
a las semillas ensayadas. Validación con semillas nuevas requiere otro acuerdo.

Resultado: cuatro continuaciones completas, 4000 épocas adicionales en 88.34 s;
0/2 éxitos por tasa. Los máximos siguen en 2; semillas 42 y 43 mantienen 30/32
y 31/32 patrones en tolerancia. La reducción a 0.001 mejora métricas secundarias
sin mejorar consistencia. Pasaron 14 pruebas y la recarga de cuatro modelos;
se verificaron 1999 archivos originales intactos. `final/` conserva la semilla
44 exitosa. La variable de hilos BLAS no llegó desde WSL a Python Windows;
detalle documentado en `comparison.md`, sin repetir corridas. Próximo trabajo
experimental: pendiente de un nuevo acuerdo con el usuario.

Estado del plan: confirmado por el usuario el 2026-11-08. La implementación comienza al invocar `assignment-step`; el presupuesto experimental se resolverá en el paso 3.

## Identidad y alcance

- Consigna: PA-004, `sources/practical-assignment/SIATP5.pdf`, p. 1, apartado 1.a; extracto `sources/extracted/PA-004.md` (rutas relativas al repositorio académico).
- Workspace: `tps_sia/tp5`, dentro del repositorio Git independiente `tps_sia`.
- Solo 1.a. Denoising (1.b) y VAE (2) quedan para planes posteriores.
- Fuente de datos: `data/font.h`, trasladada por pedido del usuario desde `C:\Users\Mateo\Downloads\font.h`, sin alterar su contenido. SHA256: `095B431BDFA2D228EEAA337FB45CD963088CC13380C189E01436E446767502F8`.
- La consigna exige representar 32 patrones de 5×7 en dos dimensiones, estudiar arquitecturas y optimización, graficar los códigos y generar una letra nueva. Solo permite recurrir a un subconjunto explicando por qué no se aprendieron todos los patrones.
- Nunca modificar originales académicos ni compilar contenido de esta solución en la wiki.

## Decisiones acordadas

Actualización 2026-11-08: tras revisar el piloto, el usuario autorizó la campaña
de ocho corridas únicas (máximo 24000 épocas) detallada en `comparison.md` y
`configs/comparison-architectures.json`. Etapa 1 compara base, ancha y profunda
con Adam 0.001 y semillas 42/43. Etapa 2 compara Adam y Momentum 0.05/0.9 sobre
la arquitectura ganadora, reutilizando Adam. Lote 32, máximo 3000 épocas por
corrida, evaluación cada 100, parada al alcanzar tolerancia. Selección por medias
entre semillas de máximo error, error medio y MSE; modelo por el mismo orden
individual dentro de la variante ganadora.

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

## Presupuesto piloto autorizado — 2026-11-08

El usuario autorizó Adam, semilla 42, tasa 0.001, lote 32, máximo 10.000 épocas, evaluación cada 100 y parada cuando los 32 patrones tengan individualmente como máximo un error. Una corrida; máximo total 10.000 épocas. La comparación queda pendiente de revisar el piloto y acordar su protocolo.

---

# Ejercicio 1.b — Denoising Autoencoder

Fecha: 2026-11-10. Estado: confirmado por el usuario en la revisión final.
La planificación no inicia implementación ni entrenamiento; continuar con
`assignment-step` para ejecutar un paso.

## Alcance y autoridad

- PA-004, p. 1, apartado 1.b.1–2: proponer y justificar una arquitectura de DAE
  sobre el mismo dataset y estudiar la eliminación de ruido a distintos niveles.
- Workspace: `tps_sia/tp5`, rama `feat/tp5`.
  Los comandos de este apartado parten de este directorio, salvo indicación.
- Fuentes académicas, relativas al repositorio del curso:
  `sources/extracted/PA-004.md`; original `sources/practical-assignment/SIATP5.pdf`;
  `wiki/topics/autoencoders.md`, Funcionamiento y Variantes y usos;
  `sources/extracted/OFF-022.md`, pp. 20–22, especialmente p. 22.
- El par de entrenamiento es entrada contaminada / objetivo limpio, según
  OFF-022 p. 22. Los niveles, presupuesto y procedimientos siguientes son
  acuerdos experimentales, no exigencias textuales de la consigna.
- Incluye código, campaña, evaluación, gráficos y documentación de 1.b.
  Excluye VAE, letras desconocidas, nueva búsqueda de 1.a y modificaciones de wiki
  u originales. Mantener los 32 caracteres de `data/font.h`.

## Acuerdos de comportamiento

1. **Referencia actualizable a 1.a.** La configuración no contiene una copia fija
   de las capas, nombre de optimizador ni tasa ganadores. Expone
   `source_selection`, inicialmente `results/final/selection.json`,
   resuelta respecto del directorio TP5. Leer allí las rutas de modelo y
   configuración respecto del TP5 de origen, según `paths_relative_to`.
   Al preparar una campaña nueva, resolver y validar la selección vigente,
   cotejar modelo/configuración/dataset y hashes, y conservar copias identificadas
   dentro de la campaña. Entrenamiento y evaluación posteriores usan esas copias.
   Cambios posteriores de 1.a solo afectan campañas nuevas; nunca mezclar dos
   selecciones en una campaña ni reemplazar su referencia al evaluar.
   No codificar la arquitectura ni el optimizador que tenía 1.a durante esta charla.
2. **Dos variantes.** `winner` usa las capas de esa selección; `latent-10` deriva
   las mismas capas sustituyendo solamente la central por 10. Ambas toman
   optimizador y tasa de la configuración fuente; conservan tanh interna y salida
   logística del contrato actual. Rechazar fuentes incompatibles con un mensaje
   explicativo, sin adaptar silenciosamente arquitectura o activaciones. Si ambas
   variantes resultan idénticas, detener preparación y revisar el protocolo.
3. **Pesos nuevos.** Entrenar todos los DAE desde cero. Los pesos de 1.a solo
   sirven para la referencia de evaluación, sin ajuste adicional.
4. **Ruido.** Invertir cada bit independientemente con probabilidad `p`. En cada
   época generar una entrada ruidosa por carácter, sorteando para cada imagen
   uniformemente un nivel entre `[0, 0.05, 0.10, 0.20]`. Los objetivos son siempre
   los originales. Regenerar niveles y máscaras en cada época; no confundir la
   probabilidad con una cantidad exacta de píxeles ni usar augmentation geométrica.
5. **Colecciones independientes.** Generar validación y evaluación final con
   semillas distintas (`20261110` y `20261111`) e independientes del entrenamiento.
   Cada colección contiene 100 versiones por carácter para cada nivel positivo
   de `[0.05, 0.10, 0.20, 0.30]`, y una versión por carácter en `p=0`:
   12.832 ejemplos por colección. Compartirlas entre variantes, semillas y
   referencias. Guardar entradas, objetivos o índices del original, niveles,
   semillas, versiones y hashes. Generación independiente no garantiza ausencia
   de coincidencias casuales; no deduplicar ni cambiar la distribución.
6. **Presupuesto.** Dos variantes × semillas `[42,43,44]`: seis corridas únicas,
   lote 32, 3000 épocas cada una, máximo total 18.000 épocas. Evaluar cada 100 y
   al inicio. Completar el presupuesto; no parar por tolerancia ni ampliar,
   reiniciar o repetir corridas automáticamente. Una interrupción debe registrarse.
   Esta aprobación define el plan; no autoriza entrenar durante su elaboración.
7. **Selección.** Para cada corrida, guardar el checkpoint que minimice el error
   binario medio en validación, promediando por igual `p=0.05,0.10,0.20`;
   desempatar por MSE medio en esos niveles y luego por época más temprana.
   Comparar variantes por el promedio de los mejores checkpoints de sus tres
   semillas usando error binario y luego MSE. Elegir una corrida representativa
   de la variante ganadora por esos mismos criterios y luego menor semilla.
   Conservar resultados de todas las semillas; no declarar ganadora una variante
   incompleta. `p=0` es diagnóstico y `p=0.30` queda para evaluación final, fuera
   del rango entrenado; ninguno participa en selección.
8. **Evaluación.** Tras fijar la selección, evaluar los seis mejores checkpoints
   y el modelo congelado de 1.a sobre la colección final. Comparar con la entrada
   ruidosa sin procesar. Para cada nivel y carácter: errores de píxeles de entrada
   y salida, reducción media de errores, fracción exacta, fracción con ≤1 error,
   máximo error y MSE. Registrar además fracciones de ejemplos que mejoran,
   empeoran o quedan iguales. Binarizar con `>=0.5`. No imponer éxito universal
   ni usar el peor ejemplo como regla de selección. Mostrar variación entre semillas
   y resultados individuales; no interpretar esto como generalización a otras letras.
9. **Entrega.** Configuraciones JSON y comandos Python reproducibles, modelos,
   manifiestos, métricas JSON/CSV, curvas por nivel y láminas original / ruido /
   AE de 1.a / DAE. Ejemplos visuales elegidos por índices fijos de la colección,
   sin seleccionar únicamente casos favorables. Informar límites y degradación
   con ruido alto, incluidos resultados negativos.

No se crea ADR: estas elecciones experimentales son locales y reversibles.
Vocabulario acordado en `../GLOSSARY.md`.

## Hechos de la implementación disponible

- `src/dataset.py` conserva los 32 patrones y su hash; `src/metrics.py` admite
  matrices `(N,35)` y ya implementa las métricas básicas.
- `src/autoencoder.py`: `validate_architecture` y `decode` restringen el código
  a 2; `train(x, epochs)` fuerza objetivo igual a entrada. Ajustar esos contratos.
- `../tp3/shared/mlp.py`, `MLP.entrenar(X,y,epocas=...)`, admite objetivos separados
  y épocas adicionales conservando pesos, RNG, historia y estado del optimizador.
  Se puede llamar una vez por época con nuevos pares; no duplicar backpropagation.
- `src/train.py` valida un esquema exacto de siete claves de 1.a. Crear flujo y
  esquema propios de denoising, preservando los comandos y configuraciones de 1.a.
- `src/explore.py` produce visualizaciones bidimensionales: conservar esa restricción
  explícita en exploración, aunque el modelo genérico admita otro ancho central.
- La referencia final de 1.a está en `results/final/` de este workspace.
  Su disponibilidad y consistencia se verifican al preparar la campaña.
- En esta shell hay `python3`, sin NumPy, y no hay `python` ni `.venv` local.
  Preparar el entorno antes de verificar código. No se ejecutaron pruebas al planificar.

## Pasos ejecutables después de confirmar el plan

Todos pendientes. Los nombres de comandos nuevos siguientes son interfaces
propuestas que deben implementarse, no comandos ya disponibles.

### B1. Referencia a la ganadora y configuración de campaña

Estado: pending. Prerrequisito: confirmación final. Entradas: selección de 1.a,
su modelo, configuración y `data/font.h`.

- Crear `configs/denoising.json` con referencia y acuerdos anteriores, y el módulo
  `src/denoise.py` con subcomando `prepare`. Validar esquema, rutas y compatibilidad
  antes de crear artefactos. Reutilizar carga de dataset y persistencia existentes.
- Preparar un directorio nuevo `results/denoising/`, congelar la referencia en
  `reference/`, derivar ambas variantes y registrar hashes y configuración resuelta.
  No sobrescribir campañas. Una nueva selección requiere un directorio nuevo.
- Verificar con dos selecciones sintéticas de distintas capas/optimizadores que
  se resuelve la referencia sin valores ganadores codificados, y que cambiar la
  fuente no cambia una campaña preparada. Rechazar hashes o dimensiones incompatibles.

Resultado: configuración, resolución y manifiesto reproducibles. Aceptación:
preparación coherente, variante derivada solo por ancho central, fuente intacta.
Comando propuesto: `python -m src.denoise prepare --config configs/denoising.json --output results/denoising`.
Referencias: PA-004 1.b.1; `src/dataset.py`, `src/autoencoder.py`, selección y
configuración referenciadas; acuerdos 1–3. Prueba: `python -m unittest discover -s tests -p "test_denoise_reference.py" -v`.

### B2. Corrupción y colecciones reproducibles

Estado: pending. Depende de B1. Entradas: dataset identificado y configuración resuelta.

- Crear `src/noise.py`, con validación de matrices binarias, probabilidades y
  semillas. Generar niveles por imagen y máscaras por píxel. Usar un generador
  separado del MLP; derivar el de entrenamiento de semilla de corrida y número
  absoluto de época para reproducir la secuencia sin depender de evaluaciones.
- Incorporar a `prepare` la generación y guardado de las dos colecciones del
  acuerdo 5. Conservar orden de caracteres e índices de realización. No ofrecer
  datos finales al selector ni a la historia de validación.
- Probar extremos `p=0` y `p=1`, reproducibilidad, objetivos intactos, formas,
  niveles admitidos, semillas independientes y asignación de originales. No usar
  tests aleatorios frágiles que exijan un porcentaje exacto de inversiones.

Resultado: colecciones NPZ y manifiestos; generador dinámico de entrenamiento.
Aceptación: 12.832 ejemplos por colección, correspondencia exacta con originales,
reproducción por configuración y hashes. Referencias: OFF-022 p. 22; PA-004 1.b.2;
`src/dataset.py`; acuerdos 4–5. Prueba: `python -m unittest discover -s tests -p "test_noise.py" -v`.

### B3. Interfaz del modelo y entrenamiento con objetivos limpios

Estado: pending. Depende de B2. Entradas: interfaz existente y variantes resueltas.

- Generalizar validación/carga/codificación/decodificación al ancho central
  especificado por la arquitectura; conservar el default y llamadas de 1.a.
  Permitir objetivos explícitos en entrenamiento con parámetro opcional, sin
  cambiar el significado de las llamadas existentes. Validar formas y finitud.
- Mantener peso y optimizador en un mismo MLP entre épocas; alimentar una nueva
  realización de ruido por época con el mismo objetivo limpio. No tocar TP3 si
  la interfaz existente basta. Evitar la pérdida de entrenamiento como sustituto
  de validación y documentar su normalización respecto de MSE.
- Guardar por separado último estado y mejor modelo, historia de validación,
  configuración y épocas. Registrar interrupciones y rechazar destinos ocupados
  al iniciar corridas; no prometer reanudación automática como parte de este alcance.
- Mantener exploración de 1.a limitada a código 2 con error claro para otro ancho.
- Probar rutas encode/decode/reconstruct y persistencia para dos anchos, objetivos
  separados, continuidad del optimizador y compatibilidad del entrenamiento limpio.
  Un flujo corto debe conservar artefactos coherentes sin exigir convergencia.

Resultado: interfaz reutilizable y entrenamiento de DAE verificable.
Referencias: OFF-022 pp. 3–6 y 22; `src/autoencoder.py`, `src/train.py`,
`src/explore.py`, `../tp3/shared/mlp.py`, `../tp3/shared/optimizers.py`; acuerdos 2–7.
Pruebas: `python -m unittest discover -s tests -p "test_autoencoder.py" -v` y
`python -m unittest discover -s tests -p "test_denoise_train.py" -v`.

### B4. Campaña y selección por validación

Estado: pending. Depende de B3 y sus pruebas aprobadas. Entradas: campaña preparada.

- Implementar subcomando `train` para las seis corridas acordadas, presupuestos
  y checkpoints cada 100 épocas. Añadir selección según acuerdo 7 y manifiestos
  de modelo/dataset/configuración, épocas, tiempo y motivo de parada.
- Probar ranking con métricas conocidas y campañas incompletas antes de entrenar.
  Ejecutar la campaña y conservar todas las corridas. Registrar cualquier fallo;
  detenerse para revisar antes de autorizar repeticiones o mayor presupuesto.
- Guardar `selection.json`, tablas y curvas de validación sin inspeccionar las
  métricas finales para tomar decisiones. No cambiar protocolo por resultados pobres.

Comando propuesto: `python -m src.denoise train --campaign results/denoising`.
Resultado: seis corridas de 3000 épocas, mejores checkpoints y selección trazable,
o interrupción explícita sin selección definitiva. Referencias: PA-004 1.b.1–2;
OFF-022 p. 22; acuerdos 6–7; módulos de B1–B3.
Prueba previa: `python -m unittest discover -s tests -p "test_denoise_selection.py" -v`.

### B5. Evaluación final, referencias y figuras

Estado: pending. Depende de B4 completo y selección congelada.

- Implementar subcomando `evaluate`; cargar los mejores checkpoints y la copia
  de 1.a sin entrenar. Aplicar todos a las mismas entradas finales y originales.
- Extender `src/metrics.py` o un módulo específico con reducción de errores y
  comparación por ejemplo; agrupar por nivel y carácter, conservando datos
  individuales. El promedio sobre niveles debe ser explícito, sin sobrerrepresentar
  niveles con más ejemplos ni confundir variación entre semillas con entre imágenes.
- Generar JSON/CSV, curvas de errores y recuperación por nivel y láminas comparables
  usando índices fijos, incluido ruido cero y 30 %. Guardar índices y hashes.
- Probar métricas con casos exactos, mejora y empeoramiento conocidos, incluidos
  umbral 0.5 y ruido cero. Recalcular desde modelos guardados y verificar consistencia.

Comando propuesto: `python -m src.denoise evaluate --campaign results/denoising`.
Resultado: métricas y figuras reproducibles sin entrenamiento. Aceptación: tres
referencias comparables, todas las semillas informadas y nivel 30 % identificado
como fuera del rango entrenado. Ninguna exigencia inventada de éxito numérico.
Referencias: PA-004 1.b.2; OFF-022 pp. 21–22; `src/metrics.py`, `src/plots.py`;
acuerdos 5, 7–9. Prueba: `python -m unittest discover -s tests -p "test_denoise_evaluation.py" -v`.

### B6. Informe, reproducción y verificación

Estado: pending. Depende de B5.

- Ampliar README para 1.b sin borrar instrucciones de 1.a y escribir
  `results/denoising/report.md`: procedencia dinámica de arquitectura y optimizador,
  justificación, protocolo, resultados, referencias comparativas, variación entre
  semillas y límites. Distinguir requisitos de la consigna, acuerdos y evidencia.
- Documentar preparación de entorno con `requirements.txt` y comandos efectivos.
  Los ejemplos usan `python` una vez activado un entorno; en esta shell puede
  crearse uno con `python3 -m venv .venv`. Instalar dependencias solo al implementar,
  con los permisos del entorno cuando sean necesarios.
- Ejecutar `python -m unittest discover -s tests -p "test_*.py" -v` y comprobar
  número positivo de pruebas. Repetir evaluación desde persistencia, sin repetir
  entrenamientos costosos. Si fue imprescindible modificar TP3, ejecutar además
  `python -m unittest discover -s shared/tests -v` desde `../tp3`.
- Verificar que PA-004 1.b.1 y 1.b.2 tengan evidencia enlazada, que originales
  conserven hashes y que resultados de 1.a no hayan sido sobrescritos. Informar
  problemas de convergencia como resultados, sin llamarlos pruebas de imposibilidad.

Aceptación: documentación y evidencia verificables, pruebas relevantes aprobadas.
Referencias: PA-004 1.b completo; OFF-022 pp. 20–22; acuerdos y artefactos B1–B5.

## Continuación de 1.b

Plan confirmado por el usuario el 2026-11-10. `assignment-step` puede ejecutar
exactamente B1 y actualizar su estado, sin iniciar B2 ni entrenar.
Conservar estados de esta sección separados del plan histórico de 1.a.
