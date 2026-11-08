# Comparación por etapas

## Continuación nocturna

La ejecución de primer plano fue detenida por pedido del usuario. Se conservan
cuatro corridas completas: base con semillas 42/43 y ancha con semillas 42/43.
La profunda con semilla 42 quedó interrumpida sin modelo utilizable; la profunda
43 y la comparación de optimizadores todavía no se ejecutaron. No hay selección
final ni demostración de nueva letra.

Para continuar desde `tp5/` en una terminal y dejar el trabajo corriendo:

```powershell
python -m src.overnight --retry-interrupted
```

Este comando **autoriza expresamente un reintento completo** de la corrida
interrumpida; archiva su intento original y conserva las cuatro corridas completas
sin entrenarlas nuevamente. Carga sus configuraciones, comprueba dataset y hash de
modelo y reutiliza sus métricas. El intento interrumpido se carga conservadoramente
como 3000 épocas consumidas: el máximo acumulado pasa de 24000 a **27000 épocas**,
incluyendo ese intento. No se ha ejecutado este reintento. Sin
`--retry-interrupted`, el comando se detiene al encontrar la corrida incompleta y
explica la decisión necesaria. Otros fallos futuros tampoco se repiten sin el
indicador; cada reintento adicional suma su propio máximo y queda archivado.

Después de completar las seis corridas de arquitectura, prepara automáticamente
`configs/comparison-optimizers.json` con la ganadora, reutiliza sus corridas Adam
y ejecuta únicamente las dos corridas Momentum pendientes. Si falta una corrida
o falla, no selecciona sobre una comparación parcial ni avanza de etapa.
Los procesos futuros guardan `checkpoint.npz` con pesos, estado del optimizador e
historia en cada evaluación. Esta continuación conserva los checkpoints pero
**reinicia** los intentos incompletos cuando se autoriza un reintento; no afirma
reanudar desde la última época. Así la contabilidad del presupuesto es explícita.

La terminal y `results/overnight.log` muestran evaluaciones y avance. Consultar
también `results/overnight-status.json`, `progress.json`, `summary.json` y las
tablas dentro de cada etapa. El comando termina con una selección rastreable o
un motivo concreto de detención; no amplía la búsqueda, valida letras ni modifica
la wiki. El modelo elegido debe revisarse después con el usuario.

El protocolo de esta campaña fue autorizado por el usuario el 2026-11-08 después
de revisar el piloto. Etapa 1: arquitecturas base `35→20→2→20→35`, ancha
`35→40→2→40→35` y profunda `35→30→15→2→15→30→35`; Adam con tasa 0.001,
semillas compartidas 42 y 43. Etapa 2: arquitectura ganadora, Adam contra Momentum
con tasa 0.05 y momentum 0.9, mismas semillas; reutilizar las dos corridas Adam.
Cada corrida tiene lote 32, máximo 3000 épocas, evaluación cada 100 y parada
cuando todos los patrones tienen como máximo un error. Son ocho corridas únicas,
como máximo 24000 épocas; el piloto anterior no forma parte de esta comparación.

Se elige arquitectura por las medias entre semillas, ordenadas por máximo error,
error medio y MSE. Entre las corridas de la variante ganadora se aplica el mismo
orden individual; los empates se resuelven por nombre de variante/identificador
de corrida. Una variante incompleta no puede ganar la comparación entre semillas.
No se adopta un subconjunto ni se amplía este presupuesto automáticamente.

Desde `tp5/`, después de instalar `requirements.txt`:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m src.compare --protocol configs/comparison-architectures.json --output results/comparison-architectures
```

La cantidad de hilos BLAS controla el costo local, sin modificar épocas, semillas
ni el algoritmo. Cada salida debe estar vacía: se conservan los resultados
anteriores. Cada corrida guarda configuración y artefactos del flujo `src.train`.
`progress.json` registra cada intento inmediatamente; `summary.json`, `runs.csv`,
`aggregates.csv` y `comparison.png` muestran resultados individuales y agregados,
éxito y costo. Las corridas fallidas e interrumpidas permanecen en las tablas con
su motivo; una interrupción impide iniciar las corridas restantes, que quedan
como `not_run`. Las fracciones de éxito usan todas las semillas previstas como
denominador. Las medias y desviaciones describen únicamente corridas completadas.

Para otra etapa se usa el mismo comando con otro protocolo JSON. Sus claves son
`stage`, `approval` (referencia a la decisión del usuario), `seeds`, `selection`
(`max_mean_mse`) y `variants`. Cada variante incluye `name`, `architecture`,
`optimizer`, `batch_size`, `max_epochs`, `evaluate_every` y `stop_on_tolerance`.
`reuse_runs` es opcional: asocia identificadores como `adam-seed-42` con los
directorios de corridas anteriores. Se exige igualdad exacta de configuración,
dataset y hash de modelo; reutilizar conserva identidad y costo de la corrida
original, pero descuenta sus épocas del presupuesto de entrenamiento nuevo.
No se reentrena una corrida reutilizada.

`selected_variant` identifica la ganadora; `selected` enlaza corrida, semilla,
configuración y modelo con hash. La selección no implica haber aprendido los
32 patrones: comprobar `all_within_tolerance`. Si no se alcanza el objetivo,
revisar la evidencia con el usuario antes de otra campaña. Esta búsqueda finita
no demuestra imposibilidad matemática ni generalización a letras desconocidas.

Pruebas cortas, sin exigir convergencia ni lanzar la campaña:

```powershell
python -m unittest discover -s tests -p "test_compare.py" -v
```
