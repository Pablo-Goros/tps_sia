import time
import pytest
from tp2.src.models.individual import Individual
from tp2.src.stopping.stopping import (
    StoppingCondition, validate_triangle_count, MAX_ALLOWED_TRIANGLES
)

def test_validate_triangle_count():
    validate_triangle_count(50)  # OK
    with pytest.raises(ValueError):
        validate_triangle_count(0)
    with pytest.raises(ValueError):
        validate_triangle_count(MAX_ALLOWED_TRIANGLES + 1)

def test_max_generations_stopping():
    cond = StoppingCondition(max_generations=10, timeout_seconds=None)
    ind = Individual.random(5)
    ind.fitness = 0.5
    stop, reason = cond.should_stop(generation=10, best_individual=ind, diversity=0.1)
    assert stop is True
    assert "generaciones" in reason.lower()

def test_timeout_stopping():
    cond = StoppingCondition(max_generations=100, timeout_seconds=0.01)
    time.sleep(0.02)
    ind = Individual.random(5)
    ind.fitness = 0.5
    stop, reason = cond.should_stop(generation=1, best_individual=ind, diversity=0.1)
    assert stop is True
    assert "timeout" in reason.lower()

def test_target_fitness_stopping():
    cond = StoppingCondition(max_generations=100, timeout_seconds=None, target_fitness=0.9)
    ind = Individual.random(5)
    ind.fitness = 0.95
    stop, reason = cond.should_stop(generation=1, best_individual=ind, diversity=0.1)
    assert stop is True
    assert "fitness objetivo" in reason.lower()

def test_content_stopping():
    cond = StoppingCondition(max_generations=100, timeout_seconds=None, content_window=3, content_delta=1e-4)
    ind = Individual.random(5)
    ind.fitness = 0.8
    # Simulate 3 generations with same fitness
    cond.should_stop(1, ind, 0.1)
    cond.should_stop(2, ind, 0.1)
    stop, reason = cond.should_stop(3, ind, 0.1)
    assert stop is True
    assert "contenido" in reason.lower()
