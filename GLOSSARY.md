# Trabajos prácticos de Sistemas de Inteligencia Artificial

Vocabulario acordado para interpretar los resultados de los trabajos prácticos.

## Autoencoders del TP5

**Patrón reconstruido dentro de tolerancia**:
Un carácter cuya reconstrucción binaria difiere del original en como máximo un píxel. La tolerancia se evalúa individualmente para cada patrón.

**Conjunto aprendido dentro de tolerancia**:
Los 32 patrones originales reconstruidos, cada uno dentro de la tolerancia acordada. No implica generalización a caracteres desconocidos.

**Candidato nuevo**:
Una imagen generada diferente de todos los patrones de entrenamiento. Su novedad no implica que sea una letra reconocible.

**Nueva letra**:
Un candidato nuevo que, tras una revisión visual, se considera interpretable como letra.

## Denoising del TP5

**Patrón limpio**:
Uno de los 32 caracteres binarios originales de `font.h`, usado como referencia para recuperar una entrada contaminada.

**Inversión de píxeles**:
Ruido que cambia cada píxel binario de 0 a 1 o de 1 a 0, independientemente, con una probabilidad dada.

**Nivel de ruido**:
Probabilidad de inversión de cada píxel. Indica la proporción esperada de píxeles alterados, no una cantidad fija de cambios en cada imagen.

**Realización de ruido**:
Una selección concreta de píxeles invertidos sobre un patrón limpio. Un mismo nivel de ruido admite distintas realizaciones.

**Colección de validación**:
Conjunto fijo de realizaciones de ruido sobre los 32 caracteres conocidos, reservado para elegir modelos durante el estudio.

**Colección de evaluación final**:
Conjunto de realizaciones de ruido generado por separado de la validación y del entrenamiento, reservado para medir resultados una vez elegidos los modelos. Evalúa recuperación de caracteres conocidos ante nuevas perturbaciones.
