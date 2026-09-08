from tp2.src.models.individual import Individual
from tp2.src.operators.selection import EliteSelection
from tp2.src.operators.survival import (
    AdditiveSurvival, ExclusiveSurvival, GenerationalGapSurvival
)

def test_additive_survival():
    sel = EliteSelection()
    strat = AdditiveSurvival(selector=sel)

    parents = [Individual.random(5) for _ in range(10)]
    for i, p in enumerate(parents):
        p.fitness = i * 0.1

    offspring = [Individual.random(5) for _ in range(8)]
    for i, o in enumerate(offspring):
        o.fitness = (i + 5) * 0.1

    next_gen = strat.form_next_generation(parents, offspring, n=10)
    assert len(next_gen) == 10
    # The best from either parents or offspring is chosen
    assert max(ind.fitness for ind in next_gen) >= 1.0

def test_exclusive_survival():
    sel = EliteSelection()
    strat = ExclusiveSurvival(selector=sel)

    parents = [Individual.random(5) for _ in range(10)]
    for p in parents:
        p.fitness = 0.2

    # Case K > N
    offspring_large = [Individual.random(5) for _ in range(15)]
    for o in offspring_large:
        o.fitness = 0.5
    next_gen = strat.form_next_generation(parents, offspring_large, n=10)
    assert len(next_gen) == 10

    # Case K <= N
    offspring_small = [Individual.random(5) for _ in range(4)]
    for o in offspring_small:
        o.fitness = 0.5
    next_gen_small = strat.form_next_generation(parents, offspring_small, n=10)
    assert len(next_gen_small) == 10

def test_generational_gap():
    sel = EliteSelection()
    strat = GenerationalGapSurvival(selector=sel, gap=0.6)

    parents = [Individual.random(5) for _ in range(10)]
    for p in parents:
        p.fitness = 0.2

    offspring = [Individual.random(5) for _ in range(10)]
    for o in offspring:
        o.fitness = 0.5

    next_gen = strat.form_next_generation(parents, offspring, n=10)
    assert len(next_gen) == 10
