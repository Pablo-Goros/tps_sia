import pytest
from tp2.src.models.triangle import Triangle
from tp2.src.models.chromosome import Chromosome
from tp2.src.models.individual import Individual

def test_triangle_random_and_copy():
    t = Triangle.random()
    assert 0.0 <= t.x1 <= 1.0
    assert 0.0 <= t.y1 <= 1.0
    assert 0 <= t.r <= 255
    assert 0 <= t.g <= 255
    assert 0 <= t.b <= 255
    assert 0 <= t.a <= 255

    t_copy = t.copy()
    assert t_copy.x1 == t.x1
    assert t_copy.r == t.r
    assert t_copy is not t

def test_triangle_serialization():
    t = Triangle(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 255, 128, 64, 200)
    d = t.to_dict()
    t2 = Triangle.from_dict(d)
    assert t2.x1 == pytest.approx(0.1, abs=1e-4)
    assert t2.r == 255
    assert t2.a == 200

def test_chromosome_and_individual():
    chrom = Chromosome.random(num_triangles=15)
    assert len(chrom) == 15
    assert len(chrom.genes) == 15

    ind = Individual(chromosome=chrom)
    assert ind.fitness is None
    ind_copy = ind.copy()
    assert len(ind_copy.chromosome) == 15
    assert ind_copy.chromosome is not ind.chromosome
