# Contexto de trabajo: TP2 de Sistemas de Inteligencia Artificial

Este repositorio implementa exclusivamente el **Ejercicio 2** del Trabajo Práctico 2: un compresor/aproximador de imágenes basado en Algoritmos Genéticos (AG). No implementar ni dedicar esfuerzo al Ejercicio 1.

## Objetivo funcional

Dada una imagen objetivo y una cantidad definida de triángulos, el programa debe generar una aproximación visual construida por la superposición de triángulos de color uniforme sobre un canvas inicial (blanco por defecto, o un color configurable).

Cada triángulo puede tener transparencia. La salida debe permitir reconstruir la imagen generada y describir sus triángulos (geometría, color y transparencia, cuando aplique).

## Modelo del problema

- Un individuo representa una imagen candidata completa.
- Su genoma representa el conjunto ordenado de triángulos que se dibujan en el canvas.
- Un gen/triángulo debe contener, como mínimo, los datos necesarios para dibujarlo: sus vértices y color; incluir alfa si se utiliza transparencia.
- La función de aptitud debe medir objetivamente la similitud entre el renderizado del individuo y la imagen objetivo. Toda elección de métrica, representación y manejo del orden/capa de los triángulos debe quedar justificada y documentada.
- La cantidad de triángulos es un parámetro del problema; no debe confundirse con un hiperparámetro del AG.

Priorizar primero imágenes pequeñas y simples (banderas, pictogramas, siluetas, símbolos o logos sencillos) para validar el motor y mantener tiempos de evaluación razonables.

## Capacidades obligatorias del motor de AG

Implementar y mantener disponibles todos estos métodos de selección:

- Elite.
- Ruleta.
- Universal.
- Boltzmann.
- Torneos: ambas variantes vistas en clase.
- Ranking.

Implementar ambas estrategias de supervivencia para crear generaciones:

- Aditiva.
- Exclusiva.

Implementar al menos dos métodos de cruza y justificar cuándo conviene cada uno:

- Un punto.
- Dos puntos.
- Uniforme.
- Anular.

Implementar al menos dos métodos de mutación y justificar cuándo conviene cada uno:

- Gen.
- MultiGen.
- Uniforme.
- No uniforme.

Definir criterios de terminación explícitos. Como mínimo, contemplar un máximo de generaciones; se pueden sumar umbrales de error, estancamiento u otros criterios defendibles.

## Restricciones técnicas

- Se permiten librerías externas para cargar, procesar y renderizar imágenes.
- No se permite usar una librería externa que implemente el Algoritmo Genético: selección, cruza, mutación, supervivencia y evolución deben ser implementación propia.
- Preservar la reproducibilidad: exponer o registrar la semilla aleatoria cuando sea viable.
- No agregar funcionalidades ajenas al ejercicio sin que aporten a la aproximación, la experimentación o la defensa del AG.

## Resultados esperados y verificabilidad

Toda implementación debe facilitar la producción de:

- La imagen aproximada resultante.
- Una enumeración/exportación de los triángulos usados y sus atributos.
- Métricas de ejecución suficientes para el análisis: fitness o error final, evolución por generación, generaciones ejecutadas y parámetros relevantes.

Al cambiar el motor, verificar que se conserve la capacidad de ejecutar una configuración pequeña de punta a punta: cargar imagen, evolucionar, generar imagen final y reportar métricas.

## Criterios para tomar decisiones de implementación

Antes de introducir o modificar un operador, evaluar y dejar clara la respuesta a estas preguntas:

- ¿Cómo se calcula el error de aproximación y en qué dirección se optimiza el fitness?
- ¿La cruza produce descendencia válida y con posibilidades razonables de mejora?
- ¿La mutación mantiene los parámetros de cada triángulo dentro de rangos válidos?
- ¿Cómo afectan el tamaño de imagen y la cantidad de triángulos al costo de evaluar una generación?
- ¿Cuál es la configuración mínima que permite comprobar el motor de AG rápidamente?

## Entrega académica

El proyecto debe poder respaldar los entregables requeridos: código fuente, `README` de ejecución y una presentación que justifique representación, aptitud, operadores, supervivencia, terminación y métricas observadas.

## Extensiones opcionales

Son opcionales y no deben desplazar los requisitos obligatorios:

- Interpretar la cantidad de triángulos como cota máxima y finalizar también al alcanzar un error mínimo.
- Soportar otros polígonos u óvalos.
- Incorporar métodos adicionales de cruza o mutación.
- Experimentar con variantes u horizontes de Algoritmos Genéticos adicionales a los vistos en clase.
