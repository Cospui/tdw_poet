from gym.envs.registration import register

register(
    id='tdw_puzzle-v0',
    entry_point='gym_tdw.envs:TdwEnv'
)

register(
    id='tdw_puzzle_proc-v0',
    entry_point='gym_tdw.envs:TdwEnv_puzzle_1_proc'
)
