# Trabajo Práctico N.º 3: Perceptrón simple y multicapa

**Sistemas de Inteligencia Artificial — ITBA**  
**2026**

## Ejercicios de validación

Estos ejercicios sirven para validar las herramientas implementadas:

- Perceptrón simple escalón.
- Perceptrón simple lineal.
- Perceptrón simple no lineal.
- Perceptrón multicapa.

No se presentan como parte de la entrega, pero se recomienda probarlos: los datos son simples y permiten comprobar mejor que los algoritmos estén bien implementados.

### Perceptrón simple escalón

Implementar la función lógica **AND** con:

```text
x = {(-1, 1), (1, -1), (-1, -1), (1, 1)}
y = {-1, -1, -1, 1}
```

### Perceptrón simple lineal

Tomar un conjunto de muestras (por ejemplo, 50) de una función lineal (por ejemplo, `y = x`) y ajustar el perceptrón a esos datos.

### Perceptrón simple no lineal

Tomar un conjunto de muestras (por ejemplo, 50) de una función no lineal (por ejemplo, `y = tanh(x)`) y ajustar el perceptrón a esos datos.

### Perceptrón multicapa

Implementar la función lógica **XOR** con:

```text
x = {(-1, 1), (1, -1), (-1, -1), (1, 1)}
y = {1, 1, -1, -1}
```

Se recomienda realizar los cálculos a mano para estas arquitecturas:

- `[2, 2, 1]`: entrada de dos elementos, una capa oculta y una capa de salida.
- `[2, 3, 2, 1]`: entrada de dos elementos, dos capas ocultas y una capa de salida.

Se puede comparar el comportamiento del perceptrón multicapa con el perceptrón simple escalón para resolver este problema.

## Ejercicio 1: TinyModel para detección de fraude

CompanyX vende artículos online y usa un modelo llamado **BigModel** para estimar la probabilidad de que una transacción sea fraudulenta. Como la inferencia con BigModel es costosa, la empresa encarga desarrollar **TinyModel**, que busca alcanzar su rendimiento con un costo mucho menor de uso y almacenamiento. Este proceso se conoce como *Knowledge Distillation*.

Se debe iterar rápidamente: implementar y evaluar un perceptrón simple lineal y uno no lineal, con una función de activación adecuada. El objetivo es estimar la probabilidad de fraude de una transacción online: `0` equivale a 0 % y `1` a 100 %. El conjunto provisto es `fraud_dataset.csv`.

### Comparación durante el aprendizaje

Comparar el perceptrón lineal con el no lineal y responder:

1. ¿Observan *underfitting*?
2. ¿Observan saturación de las capacidades?
3. Según el potencial de aprendizaje de cada perceptrón, ¿cuál seleccionarían para estudiar su generalización?

**Aclaración:** el estudio de aprendizaje se realiza utilizando todas las muestras del conjunto de datos.

### Estudio de generalización

Una vez seleccionado uno de los perceptrones, realizar un estudio de generalización y responder:

1. ¿Qué métricas de evaluación seleccionaron y por qué?
2. ¿Qué estrategia utilizaron para manipular el conjunto de datos durante la generalización? ¿Cómo creen que se elige el mejor conjunto de entrenamiento?
3. ¿Cuál es el mejor modelo que obtuvieron para presentar al cliente? Además de un modelo más pequeño, CompanyX solicita una recomendación del umbral de detección de fraude.

### Opcionales

- **Práctico:** utilizar otra función de activación en el perceptrón simple no lineal, como ReLU. ¿Qué efecto tiene sobre las conclusiones anteriores?
- **Teórico:** el conjunto provisto contiene distintos *features*. ¿Qué otros *features* se pueden construir con la información disponible para ayudar al modelado? ¿Cuáles se podrían descartar?
- **Teórico:** investigar el concepto de calibración. ¿Por qué sería apropiado analizar y ajustar la calibración en este caso?

No comenzar los ejercicios adicionales antes de implementar los puntos necesarios para el trabajo práctico. Su objetivo es ampliar las ideas centrales del TP; quienes quieran profundizar pueden abordarlos por fuera de la materia.

### Exploración del conjunto de datos

Antes de trabajar con los datos, explorarlos: ¿qué dice la documentación de cada columna?, ¿en qué rangos se mueven las columnas?, ¿de qué está compuesto el conjunto?, ¿los datos están limpios?

## Ejercicio 2: clasificación de dígitos con un perceptrón multicapa

CompanyX encarga desarrollar un modelo que detecte automáticamente dígitos escritos a mano para acelerar la digitalización de datos de distribución. Usar un perceptrón multicapa para clasificar los dígitos del `0` al `9`.

Los datos provistos son `digits.csv` para el aprendizaje y `digits_test.csv` para estudiar la generalización. Responder:

1. ¿Cómo se evalúa el desempeño del sistema?
2. ¿Qué variantes se prueban para encontrar la solución?

Como mínimo, analizar:

- Variantes de tasa de aprendizaje.
- Variantes de arquitectura.
- Variantes de mecanismos de optimización.

Estos son los aspectos básicos a explorar; también es razonable variar otros hiperparámetros.

**Aclaración:** usar `digits.csv` tanto para ajustar parámetros como hiperparámetros. Considerar `digits_test.csv` como si el modelo se pusiera en producción y se observara su comportamiento con datos del «mundo real». La misma indicación aplica al Ejercicio 3.

## Ejercicio 3: dígitos con `more_digits.csv`

Los resultados de la primera iteración no fueron satisfactorios. CompanyX solicita alcanzar una *accuracy* mayor o igual al 98 %. Durante el desarrollo se recolectaron nuevos datos, disponibles en `more_digits.csv`.

Responder:

1. ¿Cuál es el mejor resultado que pudieron obtener con este nuevo conjunto de datos?
2. ¿Qué técnicas utilizaron para mejorar el rendimiento respecto del caso anterior?
3. Además de sus propias técnicas, ¿existen otros factores que influyeron en el cambio de rendimiento entre este ejercicio y el anterior?

### Opcionales para los ejercicios 2 y 3

- **Práctico:** ¿se puede afirmar que el modelo es robusto al ruido? ¿Cómo reacciona frente a distintos grados de ruido? Para comprobarlo, tomar la mejor solución y agregar ruido a las muestras del conjunto de generalización. Hay muchas formas de modelar ruido; por ejemplo, se puede usar ruido gaussiano, común en imágenes digitales.
- **Práctico:** ¿cómo distingue la red neuronal los dígitos y qué ocurre internamente? Estas preguntas pertenecen al área de interpretabilidad. Se pueden explorar métodos de atribución.

No comenzar los ejercicios adicionales antes de implementar los puntos necesarios para el trabajo práctico. Su objetivo es ampliar las ideas centrales del TP; quienes quieran profundizar pueden abordarlos por fuera de la materia.

## Recomendaciones de implementación

Se recomienda que el trabajo práctico considere estas funcionalidades:

- Operaciones matriciales para mejorar el rendimiento.
- Reporte del progreso mientras corren los modelos.
- Configuración extensible que permita almacenar y revisar parámetros.
- Métodos para guardar y cargar modelos. Esto permite continuar entrenando un modelo que llevó mucho tiempo, junto con su configuración extensible, sin empezar desde cero.
- Separar el almacenamiento de información de cada experimento (pérdida por época, hiperparámetros, velocidad de ejecución, etc.) de su análisis (gráficos, tablas, etc.), de forma similar al enfoque usado en simulación de sistemas.

Estas son recomendaciones para tener en cuenta a medida que avanza el trabajo, no necesariamente desde el principio. Pueden ahorrar tiempo una vez que las implementaciones sean correctas y estén revisadas, y se necesite realizar varios experimentos.
