import pytest
from tp2.src.models.individual import Individual
from tp2.src.operators.selection import (
    EliteSelection, RouletteSelection, UniversalSelection,
    RankingSelection, BoltzmannSelection, DeterministicTournamentSelection,
    ProbabilisticTournamentSelection, get_selection_operator
)

@pytest.fixture
def sample_population():
    pop = []
    for f in [0.1, 0.3, 0.8, 0.5, 0.9]:
        ind = Individual.random(num_triangles=5)
        ind.fitness = f
        pop.append(ind)
    return pop

def test_elite_selection(sample_population):
    elite = EliteSelection()
    selected = elite.select(sample_population, k=3)
    assert len(selected) == 3
    # El mejor tiene fitness 0.9
    assert selected[0].fitness == 0.9

def test_roulette_selection(sample_population):
    roulette = RouletteSelection()
    selected = roulette.select(sample_population, k=10)
    assert len(selected) == 10
    for ind in selected:
        assert ind.fitness is not None

def test_universal_selection(sample_population):
    universal = UniversalSelection()
    selected = universal.select(sample_population, k=10)
    assert len(selected) == 10

def test_ranking_selection(sample_population):
    ranking = RankingSelection()
    selected = ranking.select(sample_population, k=5)
    assert len(selected) == 5

def test_boltzmann_selection(sample_population):
    boltzmann = BoltzmannSelection(t0=1.0, tc=0.01, k_rate=0.05)
    selected = boltzmann.select(sample_population, k=5, generation=10)
    assert len(selected) == 5

def test_deterministic_tournament(sample_population):
    tourn = DeterministicTournamentSelection(m=3)
    selected = tourn.select(sample_population, k=6)
    assert len(selected) == 6

def test_probabilistic_tournament(sample_population):
    prob_tourn = ProbabilisticTournamentSelection(threshold=0.8)
    selected = prob_tourn.select(sample_population, k=6)
    assert len(selected) == 6

def test_factory_invalid_selection():
    with pytest.raises(ValueError):
        get_selection_operator("desconocido")
