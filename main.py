import numpy as np
import threading
import random
from tabulate import tabulate

world_size = np.array([2 ** 3, 2 ** 3])
max_speed = 5
fps = 1
drone_location = np.array([world_size[0] / 2, world_size[1] / 2], int)
target_location = np.array([0, 0])


def zig_zag(curr_loc):
    first_location = curr_loc[0] == 0 and curr_loc[1] == 0
    even_col = curr_loc[1] == 0 or curr_loc[1] % 2 == 0
    first_row = curr_loc[0] == 0
    last_row = curr_loc[0] == world_size[0] - 1
    last_cell = curr_loc[0] == 0 and curr_loc[1] == world_size[0] - 1

    down = [1, 0]
    up = [-1, 0]
    right = [0, 1]
    left = [0, -1]
    reset = [-(world_size[0] - 1), -(world_size[1] - 1)]

    if (last_cell):
        return reset

    if (first_location):
        return down

    if (last_row and even_col):
        return right

    if (first_row and not even_col):
        return right

    if (even_col):
        return down

    if (not even_col):
        return up


def random_move(curr_loc):
    x_speed = random.randint(-1 * max_speed, max_speed)
    y_speed = random.randint(-1 * max_speed, max_speed)
    return np.array([x_speed, y_speed])


def check_boundries(loc):
    # check player in world boundaries
    loc[0] = max(0, loc[0])
    loc[0] = min(loc[0], world_size[0] - 1)
    loc[1] = max(0, loc[1])
    loc[1] = min(loc[1], world_size[1] - 1)
    return loc


def rnn(d_loc, t_loc):
    # TODO: change with rnn
    return random_move(d_loc)


def print_world():
    world = np.zeros(world_size)
    world[target_location[0], target_location[1]] = 1
    world[drone_location[0], drone_location[1]] = 2
    print(tabulate(world))
    print('\n')


def main():
    global target_location
    global drone_location

    target_speed_vector = zig_zag(target_location)
    target_location = check_boundries(target_location + target_speed_vector)
    drone_speed_vector = rnn(drone_location, target_location)
    drone_location = check_boundries(drone_location + drone_speed_vector)

    print_world()

    threading.Timer(1.0 / fps, main).start()


main()
