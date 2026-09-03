import numpy as np
from tp2.src.models.triangle import Triangle
from tp2.src.models.individual import Individual
from tp2.src.engine.renderer import Renderer
from tp2.src.engine.fitness import FitnessEvaluator

def test_renderer_outputs():
    renderer = Renderer(width=64, height=64)
    tri = Triangle(0.1, 0.1, 0.9, 0.1, 0.5, 0.9, 255, 0, 0, 180)
    arr = renderer.render_array([tri])
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8

def test_fitness_evaluator():
    evaluator = FitnessEvaluator("tp2/ejemplos/japon.png", eval_size=(32, 32))
    ind = Individual.random(num_triangles=10)
    fitness, mse = evaluator.evaluate(ind)

    assert ind.fitness is not None
    assert ind.mse is not None
    assert 0.0 < fitness <= 1.0
    assert mse >= 0.0

def test_plot_metrics(tmp_path):
    import os
    from tp2.src.visualization import plot_metrics
    csv_file = tmp_path / "metricas.csv"
    with open(csv_file, "w") as f:
        f.write("generacion,mejor_fitness,promedio_fitness,peor_fitness,mejor_mse,diversidad,segundos_transcurridos\n")
        f.write("0,0.5,0.4,0.3,5000.0,0.01,0.1\n")
        f.write("1,0.6,0.5,0.4,4000.0,0.008,0.2\n")

    plot_file = tmp_path / "grafico.png"
    out = plot_metrics(str(csv_file), str(plot_file))
    assert os.path.exists(out)
