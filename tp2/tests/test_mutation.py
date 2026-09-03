from tp2.src.models.individual import Individual
from tp2.src.operators.mutation import (
    SingleGeneMutation, LimitedMultiGeneMutation,
    UniformMultiGeneMutation, NonUniformMutation
)

def test_single_gene_mutation():
    ind = Individual.random(num_triangles=10)
    ind.fitness = 0.5
    mut = SingleGeneMutation(pm=1.0)
    mut.mutate(ind)
    assert ind.fitness is None  # invalidates cached fitness

def test_limited_multigene_mutation():
    ind = Individual.random(num_triangles=10)
    ind.fitness = 0.5
    mut = LimitedMultiGeneMutation(pm=1.0, max_genes=3)
    mut.mutate(ind)
    assert ind.fitness is None

def test_uniform_multigene_mutation():
    ind = Individual.random(num_triangles=10)
    ind.fitness = 0.5
    mut = UniformMultiGeneMutation(pm=1.0)
    mut.mutate(ind)
    assert ind.fitness is None

def test_non_uniform_mutation():
    ind = Individual.random(num_triangles=10)
    ind.fitness = 0.5
    mut = NonUniformMutation(pm=1.0, b=2.0)
    mut.mutate(ind, generation=50, max_generations=100)
    assert ind.fitness is None
