from gym.envs.registration import register

register(
    id='reacher2D-v0',
    entry_point='reach2D.envs:ReacherBulletEnv',
)