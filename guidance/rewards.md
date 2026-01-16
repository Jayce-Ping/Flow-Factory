# Reward Model Guidance

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Table of Contents

- [Reward Model Types](#reward-model-types)
- [Built-in Reward Models](#built-in-reward-models)
- [Using Built-in Reward Models](#using-built-in-reward-models)
- [Creating Custom Reward Models](#creating-custom-reward-models)
  - [Pointwise Reward Model](#pointwise-reward-model)
  - [Groupwise Reward Model](#groupwise-reward-model)
- [Multi-Reward Training](#multi-reward-training)
- [Decoupling Training and Evaluation Reward Models](#decoupling-training-and-evaluation-reward-models)

## Reward Model Types

Flow-Factory supports two paradigms for computing rewards:

| Type | Description |
|------|-------------|
| **Pointwise** | Computes independent scores for each sample |
| **Groupwise** | Computes rewards that requires all samples of a group|

**Pointwise** models evaluate each sample independently, returning absolute scores (e.g., PickScore, CLIP similarity).

**Groupwise** models evaluate all samples in a group together, enabling rewards that depend on how a sample compares to others in the same group.

## Built-in Reward Models

| Name | Type | Description | Reference |
|------|------|-------------|-----------|
| `PickScore` | Pointwise | CLIP-based aesthetic scoring | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |
| `CLIP` | Pointwise | Image-text cosine similarity | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) |
| `PickScoreRank` | Groupwise | Ranking-based reward using PickScore | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |

## Using Built-in Reward Models

Simply specify the reward model in your config file:

```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    dtype: "bfloat16"
    device: "cuda"
    batch_size: 16
```

For single reward, you can also use the shorthand format:

```yaml
rewards:
  name: "aesthetic"
  reward_model: "PickScore"
  batch_size: 16
```

## Creating Custom Reward Models

### Pointwise Reward Model

Pointwise models receive batches of size `batch_size` and compute independent scores.
```python
# src/flow_factory/rewards/my_reward.py
from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator
from typing import Optional, List
from PIL import Image
import torch

class MyPointwiseReward(PointwiseRewardModel):
    """Custom pointwise reward model."""
    
    required_fields = ("prompt", "image")  # Declare required inputs
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Available: self.config, self.device, self.dtype, self.accelerator
        # Initialize your model here
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    ) -> RewardModelOutput:
        # Input length equals self.config.batch_size
        rewards = torch.zeros(len(prompt), device=self.device)
        return RewardModelOutput(rewards=rewards)
```

### Groupwise Reward Model

Groupwise models receive the entire group at once and handle batching internally.
```python
# src/flow_factory/rewards/my_reward.py
from flow_factory.rewards import GroupwiseRewardModel, RewardModelOutput
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator
from typing import Optional, List
from PIL import Image
import torch

class MyGroupwiseReward(GroupwiseRewardModel):
    """Custom groupwise reward model with ranking."""
    
    required_fields = ("prompt", "image")
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Initialize your scoring model here
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    ) -> RewardModelOutput:
        # Input length equals group_size (NOT batch_size)
        # Handle batching internally using self.config.batch_size
        group_size = len(prompt)
        
        # Example: compute scores in batches, then rank
        all_scores = []
        for i in range(0, group_size, self.config.batch_size):
            batch_scores = self._score_batch(
                prompt[i:i + self.config.batch_size],
                image[i:i + self.config.batch_size],
            )
            all_scores.append(batch_scores)
        
        raw_scores = torch.cat(all_scores, dim=0)
        
        # Convert to rank-based rewards: [0, 1, ..., n-1] / n
        ranks = raw_scores.argsort().argsort()
        rewards = ranks.float() / group_size
        
        return RewardModelOutput(rewards=rewards)
```

**Key Differences:**

| Aspect | Pointwise | Groupwise |
|--------|-----------|-----------|
| Input size | `batch_size` samples | `group_size` samples |
| Batching | Handled by trainer | Handled internally |
| Reward semantics | Absolute scores | Relative/ranking-based |

### Register and Use
```yaml
rewards:
  - name: "custom"
    reward_model: "flow_factory.rewards.MyPointwiseReward"  # Full Python path
    batch_size: 16
```

## Multi-Reward Training

Train with multiple reward signals by adding entries to `rewards`:
```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    weight: 1.0
    batch_size: 16
    
  - name: "text_align"
    reward_model: "CLIP"
    weight: 0.5
    batch_size: 32
```

**Automatic deduplication:** Identical configurations share the same model instance to save GPU memory.

```yaml
rewards:
  - name: "aesthetic_1"
    reward_model: "PickScore"
    batch_size: 16
    
  - name: "aesthetic_2"
    reward_model: "PickScore"  # Same config → reuses model above
    batch_size: 16
```

## Decoupling Training and Evaluation Reward Models

Use different reward models for training and evaluation:

```yaml
# Training rewards
rewards:
  - name: "fast_score"
    reward_model: "PickScore"
    batch_size: 32

# Evaluation rewards (optional)
eval_rewards:
  - name: "hps"
    reward_model: "my_rewards.HPSv2RewardModel"
    batch_size: 8
```

If `eval_rewards` is not specified, training rewards are reused for evaluation.

**Use cases:**
- Train with fast model, evaluate with slower but more accurate model
- Cross-model evaluation to detect overfitting