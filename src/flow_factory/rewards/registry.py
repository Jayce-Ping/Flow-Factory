# src/flow_factory/rewards/registry.py
from typing import Type, Dict, Any
import importlib

# 1. The Registry Storage
_REWARD_MODEL_REGISTRY: Dict[str, str] = {
    'PickScore': 'flow_factory.rewards.pick_score.PickScoreRewardModel',
}

# 2. The Decorator (For easy registration of new internal models)
def register_reward_model(name: str):
    def decorator(cls):
        _REWARD_MODEL_REGISTRY[name] = f"{cls.__module__}.{cls.__name__}"
        return cls
    return decorator

# 3. The Dynamic Loader Utility
def get_reward_model_class(identifier: str) -> Type:
    """
    Resolves a class from a registered name OR a python dotted path.
    Example: 
      - 'PickScore' -> returns PickScoreRewardModel class
      - 'my_lib.rewards.CustomReward' -> imports and returns CustomReward class
    """
    # a) Check Registry
    class_path = _REWARD_MODEL_REGISTRY.get(identifier, identifier)
    
    # b) Dynamic Import
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not load reward model '{identifier}'. "
                          f"Ensure it is a valid python path (module.submodule.ClassName) "
                          f"or a registered name. Error: {e}")