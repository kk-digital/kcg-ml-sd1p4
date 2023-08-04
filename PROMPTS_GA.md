# Prompts Genetic Algorithm

Documentation for script at `./scripts/prompts_ga.py`.

The script generates text prompt phrases, which are used to compute prompt
embeddings to initialize a genetic algorithm population. The fitness function is
chad score.

Example Usage:

``` shell
python scripts/prompts_ga.py \
  --generations 2 \
  --mutation_probability=0.05 \
  --crossover_type = single_point \
  --keep_elitism 0 \
  --mutation_type random
```

## Available Selection Operators

- "sss": Steady state selection
- "rws": Roulette wheel selection
- "sus": Stochastic universal selection
- "random": Random selection
- "tournament": Tournament selection
- "rank": Rank selection
                    
## Available Mutation Operators

- "random": Random mutation
- "swap": Swap mutation
- "scramble": Scramble mutation
- "inversion": Inversion mutation
- "adaptive": Adaptive mutation

## Available Crossover Operators

- "single_point": Single point crossover
- "two_points": Two points crossover
- "uniform": Uniform crossover
- "scattered": Scattered crossover
