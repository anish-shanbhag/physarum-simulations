from slime import simulate

simulate(
    num_agents=int(5e4),
    width=800,
    height=1200,
    start_positions=[[120, 160], [560, 1100], [275, 545]],
    image_name="love.png",
    animation_name="animation1.mp4",
    animation_fps=60,
    animation_seconds=20,
    cmap=[
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0.5, 0.5, 0.5, 0.5],
            [0, 0.2, 0.5, 1],
            [0.25, 0.6, 1, 1],
            [0, 1, 1, 1],
        ],
        [0, 0.05, 0.1, 0.3, 0.5, 1],
    ],
)

simulate(
    num_agents=int(1e5),
    width=800,
    height=1200,
    start_positions=[[120, 160], [640, 850], [275, 545], [650, 250]],
    image_name="maze.png",
    animation_name="animation2.mp4",
    animation_fps=60,
    animation_seconds=20,
    cmap=[
        [[0, 0, 0, 0], [0.12, 0.25, 0.216, 1], [0.6, 0.95, 0.78, 1]],
        [0, 0.2, 1],
    ],
)

simulate(
    num_agents=int(9e5),
    width=800,
    height=1200,
    start_positions=[[400, 600]],
    image_name="empty.png",
    animation_name="animation3.mp4",
    animation_fps=60,
    animation_seconds=20,
    cmap=[
        [[0, 0, 0, 0], [0.9, 0.9, 0, 1], [0.6, 0.3, 0, 1], [1, 1, 0.3, 1]],
        [0, 0.3, 0.6, 1],
    ],
)

simulate(
    num_agents=int(4e5),
    width=800,
    height=1200,
    start_positions=[
        [300, 300],
        [500, 800],
    ],
    image_name="empty.png",
    animation_name="animation4.mp4",
    animation_fps=60,
    animation_seconds=20,
    cmap=[
        [[0, 0, 0, 0], [0.9, 0, 0.5, 1], [0.6, 0.3, 0, 1], [0.5, 0.2, 1, 1]],
        [0, 0.3, 0.6, 1],
    ],
)

simulate(
    num_agents=int(5e6),
    width=1000,
    height=1800,
    start_positions=[
        [100, 100],
        [900, 1700],
        [100, 1700],
        [900, 100],
    ],
    image_name="swirl.png",
    animation_name="animation5.mp4",
    animation_fps=200,
    animation_seconds=12,
    cmap=[
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0.2, 1],
        ],
        [0, 0.05, 0.3, 1],
    ],
)

simulate(
    num_agents=int(1e5),
    width=800,
    height=1200,
    start_positions=[[230, 140], [515, 1030]],
    image_name="eps.png",
    animation_name="animation6.mp4",
    animation_fps=90,
    animation_seconds=16,
    cmap=[
        [
            [0, 0, 0, 0],
            [0.25, 0.9, 0.8, 1],
            [1, 0.5, 0, 1],
            [1, 0, 0.5, 1],
        ],
        [0, 0.25, 0.6, 1],
    ],
)
