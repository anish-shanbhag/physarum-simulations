import math

from matplotlib import image
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from vispy import app, scene
from vispy.color.colormap import Colormap
from vispy.gloo.util import _screenshot

import imageio


def simulate(
    num_agents: int,
    width: int,
    height: int,
    start_positions: list[list[int]],
    image_name: str,
    animation_name: str,
    animation_fps: int,
    animation_seconds: float,
    cmap: list[list],
    speed: float = 2,
):
    maze_image = image.imread(image_name)
    maze = 1 - np.array(
        np.minimum(
            np.ones((width, height)),
            maze_image[:width, :height, 0],
            casting="unsafe",
        ),
        dtype=np.bool8,
    )

    sensor_angle = np.pi / 3
    random_turn_chance = 0.0
    turn_angle = np.pi / 8
    sensor_offset = 8
    start_threshold = 0
    zeros = np.zeros((width, height), dtype=np.float16)
    grid = zeros.copy()

    agents = np.multiply(
        np.random.random((num_agents, 3)), np.array([30, 30, 2 * math.pi])
    )
    for agent in agents:
        group = np.random.randint(0, len(start_positions))
        agent[0] += start_positions[group][0]
        agent[1] += start_positions[group][1]

    agent_threadsperblock = 1024
    agent_blockspergrid = (
        len(agents) + (agent_threadsperblock - 1)
    ) // agent_threadsperblock

    grid_threadsperblock = (8, 8)
    grid_blockspergrid = (
        math.ceil(len(grid) / grid_threadsperblock[0]),
        math.ceil(len(grid[0]) / grid_threadsperblock[0]),
    )

    @cuda.jit
    def update(rng_states, grid, agents, sensor_offset, i, maze):
        pos = cuda.grid(1)
        agent = agents[pos]

        # Motor stage
        new_x = agent[0] + math.cos(agent[2]) * (
            speed if i > start_threshold else 10 * speed
        )
        new_y = agent[1] + math.sin(agent[2]) * (
            speed if i > start_threshold else 10 * speed
        )

        is_inside = lambda x, y: maze[int(x)][int(y)]

        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
            agent[2] = xoroshiro128p_uniform_float32(rng_states, pos) * 2 * math.pi
            return
        elif not is_inside(new_x, new_y):
            agent[2] = xoroshiro128p_uniform_float32(rng_states, pos) * 2 * math.pi
        else:
            agent[0] = new_x
            agent[1] = new_y

        grid[int(new_x), int(new_y)] = min(10, grid[int(new_x), int(new_y)] + 5)

        if i < start_threshold:
            return
        if xoroshiro128p_uniform_float32(rng_states, pos) < random_turn_chance:
            agent[2] = xoroshiro128p_uniform_float32(rng_states, pos) * 2 * math.pi
            return

        # Sensor stage
        left_angle = agent[2] - sensor_angle
        right_angle = agent[2] + sensor_angle
        left_x = agent[0] + math.cos(left_angle) * sensor_offset
        left_y = agent[1] + math.sin(left_angle) * sensor_offset
        right_x = agent[0] + math.cos(right_angle) * sensor_offset
        right_y = agent[1] + math.sin(right_angle) * sensor_offset
        front_x = agent[0] + math.cos(agent[2]) * sensor_offset
        front_y = agent[1] + math.sin(agent[2]) * sensor_offset
        new_turn_angle = turn_angle  # * xoroshiro128p_uniform_float32(rng_states, pos)
        if (
            left_x < 0
            or left_x >= width
            or left_y < 0
            or left_y >= height
            or not is_inside(left_x, left_y)
        ):
            left = -2
        else:
            left = grid[int(left_x), int(left_y)]
        if (
            right_x < 0
            or right_x >= width
            or right_y < 0
            or right_y >= height
            or not is_inside(right_x, right_y)
        ):
            right = -2
        else:
            right = grid[int(right_x), int(right_y)]
        if (
            front_x < 0
            or front_x >= width
            or front_y < 0
            or front_y >= height
            or not is_inside(front_x, front_y)
        ):
            front = -2
        else:
            front = grid[int(front_x), int(front_y)]
        if front > left and front > right:
            return
        elif front < left and front < right:
            if xoroshiro128p_uniform_float32(rng_states, pos) < 0.5:
                agent[2] -= new_turn_angle
            else:
                agent[2] += new_turn_angle
        elif left < right:
            agent[2] += new_turn_angle
        elif right < left:
            agent[2] -= new_turn_angle

    @cuda.jit
    def blur(grid):
        x, y = cuda.grid(2)

        size = 1
        total = 0
        for i in range(x - size, x + size + 1):
            for j in range(y - size, y + size + 1):
                if i < 0 or i >= width or j < 0 or j >= height:
                    continue
                total += grid[i, j]
        result = total / (size * 2 + 1) ** 2 * 0.9
        grid[x][y] = result

    rng_states = create_xoroshiro128p_states(
        agent_threadsperblock * agent_blockspergrid, seed=1
    )

    canvas = scene.SceneCanvas(keys="interactive")
    canvas.size = height, width
    canvas.show()

    # Set up a viewbox to display the image
    view = canvas.central_widget.add_view()

    bg = scene.visuals.Image(
        maze_image,
        parent=view.scene,
        interpolation="nearest",
        cmap=Colormap([[0, 0, 0, 0], [1, 1, 1, 1]]),
    )
    bg.set_gl_state("translucent", depth_test=False)

    slime_image = scene.visuals.Image(
        np.vstack((grid, grid, grid, 0.5 * np.ones((width, height)))),
        interpolation="nearest",
        parent=view.scene,
        cmap=Colormap(*cmap),
    )
    slime_image.set_gl_state("translucent", depth_test=False)

    @canvas.events.key_press.connect
    def on_key_press(event):
        nonlocal sensor_offset
        if event.key == "Left":
            sensor_offset -= 1
        elif event.key == "Right":
            sensor_offset += 1

    i = 0
    writer = imageio.get_writer(animation_name, fps=animation_fps)
    frames = animation_seconds * animation_fps

    def render(ev):
        nonlocal grid, zeros, slime_image, sensor_offset, i
        i += 1
        if i > frames:
            writer.close()
            app.quit()
            return
        writer.append_data(_screenshot())
        update[agent_blockspergrid, agent_threadsperblock](
            rng_states, grid, agents, int(sensor_offset), i, maze
        )
        blur[grid_blockspergrid, grid_threadsperblock](grid)
        slime_image.set_data(grid)
        slime_image.update()

    timer = app.Timer()
    timer.connect(render)
    timer.start(0)

    app.run()
