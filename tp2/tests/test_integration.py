import os
import shutil
from tp2.src.config import Config
from tp2.src.engine.fitness import FitnessEvaluator
from tp2.src.operators.selection import EliteSelection, RouletteSelection
from tp2.src.operators.crossover import TwoPointCrossover
from tp2.src.operators.mutation import UniformMultiGeneMutation
from tp2.src.operators.survival import AdditiveSurvival
from tp2.src.stopping.stopping import StoppingCondition
from tp2.src.metrics.tracker import MetricsTracker
from tp2.src.engine.genetic_algorithm import GeneticAlgorithm

def test_full_pipeline_run():
    out_dir = "tp2/tests/salidas_test"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    evaluator = FitnessEvaluator("tp2/ejemplos/japon.png", eval_size=(32, 32))
    parent_sel = RouletteSelection()
    surv_sel = EliteSelection()
    surv_strat = AdditiveSurvival(selector=surv_sel)
    crossover_op = TwoPointCrossover(pc=0.9)
    mutation_op = UniformMultiGeneMutation(pm=0.2)
    stopping = StoppingCondition(max_generations=5, timeout_seconds=10.0)
    tracker = MetricsTracker(output_dir=out_dir)

    ga = GeneticAlgorithm(
        evaluator=evaluator,
        parent_selector=parent_sel,
        crossover_operator=crossover_op,
        mutation_operator=mutation_op,
        survival_strategy=surv_strat,
        stopping_condition=stopping,
        tracker=tracker,
        population_size=10,
        num_offspring=10,
        num_triangles=8,
        snapshot_interval=2
    )

    results = ga.run()

    assert results["generations_completed"] == 5
    assert os.path.exists(results["best_image_path"])
    assert os.path.exists(results["triangles_json_path"])
    assert os.path.exists(results["metrics_csv_path"])

    # Limpieza
    shutil.rmtree(out_dir)
