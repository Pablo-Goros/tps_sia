# 01: Inspeccionar el dataset original

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** done

## What to build

Un comando ejecutable desde el espacio de trabajo del TP5 que cargue el archivo original de caracteres, valide su contenido y genere una lámina con los 32 patrones identificados. La entrega permite comprobar los datos antes de entrenar e incluye instalación, instrucciones y pruebas del comportamiento público.

Leer la matriz `Font3` sin ejecutar el header ni transcribir los caracteres. Conservar los siete renglones de cada patrón y representar cada entero como cinco bits del más significativo al menos significativo, de izquierda a derecha. Mantener el orden original de los códigos `0x60` a `0x7f`, con una etiqueta legible para DEL. Exponer los datos como imágenes `(32,7,5)` y vectores flotantes `(32,35)` aplanados por filas.

El original ya está disponible y debe conservarse intacto. Las dependencias serán compatibles con NumPy y Matplotlib del TP3. Documentar el comando efectivo y el destino de la lámina, sin requerir conocimiento del entorno del agente.

Contexto: [plan confirmado, paso 1](../../../../tp5/assignment_plan.md); criterios de aceptación 1–3 de la especificación.

## Acceptance criteria

- [x] Desde el TP5 se puede ejecutar el comando documentado y obtener una lámina legible de los 32 patrones originales, con etiquetas y orientación correctas.
- [x] El cargador obtiene exactamente 32 patrones, cada uno con siete enteros entre 0 y 31, y produce las formas `(32,7,5)` y `(32,35)` con valores binarios.
- [x] Se ignoran los comentarios al leer la matriz, sin incorporar sus códigos hexadecimales como píxeles ni ejecutar contenido del archivo.
- [x] Entradas incompletas, cantidades incorrectas de filas o valores fuera de rango producen errores comprensibles.
- [x] Una comprobación pública con patrones asimétricos detecta reflejos o inversiones del orden; verificar solo las dimensiones no basta.
- [x] El archivo original conserva el hash registrado en el plan.
- [x] El README explica instalación y reproducción; las pruebas relevantes se ejecutan mediante un comando documentado y verifican la carga y el resultado observable.

## Blocked by

None
