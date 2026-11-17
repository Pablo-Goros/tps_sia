# Experimento local: extensión de 1.b de 3000 a 6000 épocas

Estado: exploratorio. NO forma parte del protocolo aprobado ni se publica en el repo.
Los resultados publicados (3000 épocas) en `results/denoising/` siguen siendo los principales.

Regla fijada ANTES de ejecutar:

- Se parte del último estado guardado (`last_model.npz`, época 3000) de las seis corridas y se
  continúa hasta la época 6000, con el ruido de la época absoluta (semilla, época), como en el
  protocolo original. Mismos niveles de entrenamiento, mismo lote, mismo optimizador.
- Validación cada 100 épocas con la misma colección y los mismos niveles de selección
  (0,05; 0,10; 0,20). El mejor checkpoint de cada corrida es el de menor puntaje en TODO el
  rango 0–6000 (si ninguna época nueva supera al mejor de 3000, se conserva el de 3000).
- Variante ganadora y corrida representativa: misma regla que el protocolo original.
- La colección final se evalúa una sola vez, después de fijar la selección. No se ajusta nada
  con ella. Se informa el resultado sea cual sea.
- Los archivos de `results/denoising/` no se modifican (se verifica por SHA256 antes y después).
- Entorno de esta extensión: Python 3.11 / NumPy 2.3.5; la campaña original usó Python 3.14.4 /
  NumPy 2.4.4. El arranque es idéntico (mismos estados publicados), pero la numérica posterior
  puede diferir levemente de la que daría el entorno original.
