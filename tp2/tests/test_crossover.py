from tp2.src.models.individual import Individual
from tp2.src.operators.crossover import TwoPointCrossover, UniformCrossover

def test_two_point_crossover():
    p1 = Individual.random(num_triangles=20)
    p2 = Individual.random(num_triangles=20)
    cross = TwoPointCrossover(pc=1.0)
    c1, c2 = cross.cross(p1, p2)

    assert len(c1.chromosome) == 20
    assert len(c2.chromosome) == 20

def test_uniform_crossover():
    p1 = Individual.random(num_triangles=20)
    p2 = Individual.random(num_triangles=20)
    cross = UniformCrossover(pc=1.0, p_swap=0.5)
    c1, c2 = cross.cross(p1, p2)

    assert len(c1.chromosome) == 20
    assert len(c2.chromosome) == 20
