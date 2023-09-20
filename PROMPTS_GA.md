# Summary

- [Prompts Genetic Algorithms](#prompts-genetic-algorithms)
   - [Pygad-based GAs](#pygad-based-gas)
      - [scripts/prompts_ga.py](#scriptsprompts_gapy)
      - [scripts/ga_bounding_box_size.py](#scriptsga_bounding_box_sizepy)
      - [scripts/ga_bounding_box_centered.py](#scriptsga_bounding_box_centeredpy)
      - [scripts/ga_filesize.py](#scriptsga_filesizepy)
   - [LEAP-based GAs](#leap-based-gas)
      - [scripts/prompts_ga_leap.py](#scriptsprompts_ga_leappy)
      - [scripts/ga_white_background_leap.py](#scriptsga_white_background_leappy)
      - [scripts/ga_bounding_box_size_leap.py](#scriptsga_bounding_box_size_leappy)
      - [scripts/ga_filesize_leap.py](#scriptsga_filesize_leappy)
   - [Pygad Configuration](#pygad-configuration)
      - [CLI args](#cli-args)
      - [Available Selection Operators](#available-selection-operators)
      - [Available Mutation Operators](#available-mutation-operators)
      - [Available Crossover Operators](#available-crossover-operators)
   - [LEAP Configuration](#leap-configuration)
   - [Troubleshooting](#troubleshooting)

# Prompts Genetic Algorithms

Documentation for these scripts:
- `scripts/prompts_ga.py`
- `scripts/prompts_ga_leap.py`
- `scripts/ga_bounding_box_size.py`
- `scripts/ga_filesize.py`
- `scripts/ga_bounding_box_centered.py`
- `scripts/ga_filesize.py`

The script generates text prompt phrases, which are used to compute prompt
embeddings to initialize a genetic algorithm population. The fitness function is
chad score. All scripts use Pygad as their GA library, with the exception of
`scripts/prompts_ga_leap.py`.

## Pygad-based GAs

### scripts/prompts_ga.py

Example Usage:

``` shell
python scripts/prompts_ga.py --generations 5 --mutation_probability 0.10 --crossover_type single_point --keep_elitism 0 --mutation_type swap --mutation_percent_genes 0.05 --population_size 5
```

### scripts/ga_bounding_box_size.py

Documentation for script at `./scripts/ga_bounding_box_size.py`.

The script generates text prompt phrases, which are used to compute prompt
embeddings to initialize a genetic algorithm population. The fitness function is located in ga/fitness_bounding_box_size.py. 
The fitness score will be 1.0 when the object occupies one-fourth of the full image.

Example Usage:

``` shell
python scripts/ga_bounding_box_size.py --generations 100 --mutation_probability 0.05 --crossover_type single_point --keep_elitism 0 --mutation_type random --mutation_percent_genes 0.05
```

### scripts/ga_bounding_box_centered.py

Documentation for script at `./scripts/ga_bounding_box_centered.py`.

The script generates text prompt phrases, which are used to compute prompt
embeddings to initialize a genetic algorithm population. The fitness function is located in ga/fitness_bounding_box_centered.py. 
The fitness score will be 1.0 when the object is centered within the image.

Example Usage:

``` shell
python scripts/ga_bounding_box_centered.py \
  --generations 100 \
  --mutation_probability 0.05 \
  --crossover_type single_point \
  --keep_elitism 0 \
  --mutation_type random \
  --mutation_percent_genes 0.05
```



### scripts/ga_latent.py

Documentation for script at `./scripts/ga_latent.py`.

The script generates random latent vectors or can generate images from random prompt to initialize a genetic algorithm population.
The fitness function is chad_score.

Example Usage:

``` shell
python scripts/ga_latent.py
```
### CLI args

```
options:
  --generations GENERATIONS
                        Number of generations to run.
  --mutation_probability MUTATION_PROBABILITY
                        Probability of mutation.
  --keep_elitism KEEP_ELITISM
                        1 to keep best individual, 0 otherwise.
  --crossover_type CROSSOVER_TYPE
                        Type of crossover operation.
  --mutation_type MUTATION_TYPE
                        Type of mutation operation.
  --mutation_percent_genes MUTATION_PERCENT_GENES
                        The percentage of genes to be mutated.
  --use_random_images USE_RANDOM_IMAGES
                        is the flag is strue, generate random latent vectors
  --steps STEPS
                        number of steps for sampler
  --device DEVICE
                        device to use  
  --num_phrases NUM_PHRASES
                        number of phrases in the random prompt generator
  --cfg_strength CFG_STRENGTH
                        cfg_strength for the generated images
  --sampler SAMPLER
                        sampler name (ddim, ddpm)
  --cheackpoint_path CHEACKPOINT_PATh
                        stable diffusion model path, used to generate images
  --image_width IMAGE_WIDTH
                        width of the generated textures
  --image_height IMAGE_HEIGHT
                        height of the generated textures
  --output OUTPUT
                        output path for the ga

```

### scripts/ga_filesize.py

Documentation for script at `./scripts/ga_filesize.py`.

The script generates text prompt phrases, which are used to compute prompt
embeddings to initialize a genetic algorithm population. The fitness function is located in ga/fitness_filesize.py. 

Example Usage:

``` shell
python scripts/ga_filesize.py \
  --generations 100 \
  --mutation_probability 0.05 \
  --crossover_type single_point \
  --keep_elitism 0 \
  --mutation_type random \
  --mutation_percent_genes 0.05
```

## LEAP-based GAs

### scripts/prompts_ga_leap.py

Example Usage:

``` shell
python scripts/prompts_ga_leap.py --generations 100
```

### scripts/ga_white_background_leap.py

Example Usage:

``` shell
python scripts/ga_white_background_leap.py --generations 100
```

### scripts/ga_bounding_box_size_leap.py

Example Usage:

``` shell
python scripts/ga_bounding_box_size_leap.py --generations 100
```

### scripts/ga_filesize_leap.py

Example Usage:

``` shell
python scripts/ga_filesize_leap.py --generations 100
```

## Pygad Configuration

### CLI args

```
options:
  --generations GENERATIONS
                        Number of generations to run.
  --mutation_probability MUTATION_PROBABILITY
                        Probability of mutation.
  --keep_elitism KEEP_ELITISM
                        1 to keep best individual, 0 otherwise.
  --crossover_type CROSSOVER_TYPE
                        Type of crossover operation.
  --mutation_type MUTATION_TYPE
                        Type of mutation operation.
  --mutation_percent_genes MUTATION_PERCENT_GENES
                        The percentage of genes to be mutated.
```

### Available Selection Operators

- "sss": Steady state selection
- "rws": Roulette wheel selection
- "sus": Stochastic universal selection
- "random": Random selection
- "tournament": Tournament selection
- "rank": Rank selection
                    
### Available Mutation Operators

- "random": Random mutation
- "swap": Swap mutation
- "scramble": Scramble mutation
- "inversion": Inversion mutation
- "adaptive": Adaptive mutation

### Available Crossover Operators

- "single_point": Single point crossover
- "two_points": Two points crossover
- "uniform": Uniform crossover
- "scattered": Scattered crossover

## LEAP Configuration

`--generations` is the only CLI arg supported by the LEAP script at the moment.
LEAP is far more robust and flexible, but requires additional work.

## Troubleshooting

If you encounter an error like this: "ImportError: libGL.so.1: cannot open shared object file: No such file or directory" 

you can resolve it by executing the following command:

```
apt-get install libgl1-mesa-glx
```
