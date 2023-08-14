
import random



def get_seed_array_from_string(seed, array_size = 1, min_value = 0, max_value = 2 ** 24 - 1):
    seed_string_array = []
    if seed != '':
        string_array = seed.split(',')
        for string in string_array:
            integer = int(string)
            seed_string_array.append(integer)

    # default seed value is random int from 0 to 2^24
    if seed == '':
        # Generate an array of random integers in the range [0, 2^24)
        seed_string_array = [random.randint(min_value, max_value) for _ in range(array_size)]

    # Convert the elements in the list to integers
    return seed_string_array