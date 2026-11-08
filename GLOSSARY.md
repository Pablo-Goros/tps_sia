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
