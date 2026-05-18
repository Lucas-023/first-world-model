Fisrt try to build a world model for Gymnsaium Box2D Car racing Game
This is inspired by David Ha and Jürgen Schmidhuber World MOdels (2018) paper with some changes in architecture


epsilon_schedule = [
            (0.02, 0.25),
            (0.05, 0.25),
            (0.10, 0.20),
            (0.20, 0.15),
            (0.35, 0.10),
            (0.60, 0.05),
        ]
for 2000 episodes, already collected 1000:
500 with eps = 0.02
500 with eps = 0.05