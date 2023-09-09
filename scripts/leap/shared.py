import random
from leap_ec.decoder import Decoder as leapDecoder
from leap_ec.problem import ScalarProblem
from leap_ec import Individual as leapIndividual

fitness_cache = {}


def cached_fitness_func(solution, loss_func):
    if tuple(solution) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution)])
    if tuple(solution) not in fitness_cache:
        fitness_cache[tuple(solution)] = loss_func(solution)
    return fitness_cache[tuple(solution)]


class Decoder(leapDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        return genome.flatten()

    def __repr__(self):
        return type(self).__name__ + "()"


class Problem(ScalarProblem):
    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        return cached_fitness_func(phenome)
        # return calculate_size_fitness(phenome)


class Individual(leapIndividual):
    def __init__(self, genome=[], decoder=None, problem=None, seed=None):
        super().__init__(genome, decoder=decoder, problem=problem)
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        self.fitness = None
        self.individual_seed = random.randint(0, 2 ** 24)

    def set_seed(self, seed):
        self.seed = seed

    def set_individual_seed(self, individual_seed):
        self.individual_seed = individual_seed

    def set_genome(self, genome):
        self.genome = genome

    def get_seed(self):
        return self.seed

    def get_individual_seed(self):
        return self.individual_seed

    def get_genome(self):
        return self.genome

    def generate_random_genome(self, gene_count):
        random.seed(self.individual_seed)
        self.genome = [random.random() for _ in range(gene_count)]

    def __str__(self):
        return f"Individual with seed: {self.seed}, individual_seed: {self.individual_seed}, genome: {self.genome}"
