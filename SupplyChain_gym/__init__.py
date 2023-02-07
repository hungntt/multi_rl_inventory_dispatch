from gym.envs.registration import register

register(
        id="DoDistEnv-v0",
        entry_point="SupplyChain_gym.envs:InventoryInputEnv",
)
