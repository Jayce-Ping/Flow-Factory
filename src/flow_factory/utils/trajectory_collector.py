# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/utils/trajectory_collector.py
"""
Trajectory Collector for Inference

Generic utility for memory-efficient tensor recording during denoising.
Supports collecting all, none, or specific timestep indices.

Produces compact storage + lightweight index map to eliminate redundant
data in multi-GPU gather operations.
"""
from typing import Union, List, Optional, Literal, Set, TypeVar
import torch


T = TypeVar('T')
TrajectoryIndicesType = Union[Literal['all'], List[int], None]


class TrajectoryCollector:
    """
    Collects tensors at specified indices during denoising trajectory.
    
    Memory-efficient alternative to storing all intermediate values.
    Produces a compact tensor + index map for O(1) position lookup,
    avoiding redundant storage and multi-GPU communication overhead.
    
    Args:
        indices: Controls which steps to record:
            - 'all': Record all steps (default behavior)
            - None: Don't record any steps (returns None)
            - List[int]: Record only at specified indices
                - Index 0: Initial state (before denoising)
                - Index i (1 to T): State after i-th denoising step  
                - Index -1: Final state (same as index T)
                - Supports negative indexing like Python lists
        total_steps: Total number of denoising steps (T)
    
    Examples:
        >>> collector = TrajectoryCollector('all', total_steps=20)
        >>> collector = TrajectoryCollector(None, total_steps=20)     # No recording
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)  # Initial + final only
        >>> collector = TrajectoryCollector([0, 10, -1], total_steps=20)  # Specific checkpoints
    
    Usage:
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)
        >>> collector.collect(initial_latents, step_idx=0)
        >>> for i in range(20):
        ...     latents = denoise_step(latents)
        ...     collector.collect(latents, step_idx=i + 1)
        >>> trajectory = collector.get_result()       # [initial, final]
        >>> index_map = collector.get_index_map()     # [0, -1, ..., -1, 1]
    """
    
    def __init__(
        self,
        indices: TrajectoryIndicesType = 'all',
        total_steps: int = 0,
    ):
        self.indices = indices
        self.total_steps = total_steps
        self._collected: List[torch.Tensor] = []
        self._collected_indices: List[int] = []
        
        # Precompute normalized indices for O(1) lookup
        self._target_indices: Optional[Set[int]] = self._normalize_indices()
    
    def _normalize_indices(self) -> Optional[Set[int]]:
        """Convert user indices to normalized positive indices."""
        if self.indices is None:
            return None
        if self.indices == 'all':
            return None  # Signal to collect all
        
        # Total positions = total_steps + 1 (initial + each step result)
        total_positions = self.total_steps + 1
        normalized = set()
        
        for idx in self.indices:
            # Handle negative indices (Python-style)
            if idx < 0:
                idx = total_positions + idx
            # Clamp to valid range
            if 0 <= idx < total_positions:
                normalized.add(idx)
        
        return normalized
    
    @property
    def is_disabled(self) -> bool:
        """Check if collection is disabled."""
        return self.indices is None
    
    @property
    def collect_all(self) -> bool:
        """Check if collecting all steps."""
        return self.indices == 'all'
    
    def should_collect(self, step_idx: int) -> bool:
        """
        Check if value should be collected at this step.
        
        Args:
            step_idx: Current position (0=initial, 1..T=after each step)
        
        Returns:
            True if value should be recorded at this position
        """
        if self.is_disabled:
            return False
        if self.collect_all:
            return True
        return step_idx in self._target_indices
    
    def collect(self, value: torch.Tensor, step_idx: int) -> None:
        """
        Conditionally collect tensor at given step.
        
        Args:
            value: Tensor to potentially store
            step_idx: Current position index
        """
        if self.should_collect(step_idx):
            self._collected.append(value)
            self._collected_indices.append(step_idx)
    
    def get_result(self) -> Optional[List[torch.Tensor]]:
        """
        Get collected tensors.
        
        Returns:
            List of collected tensors, or None if disabled
        """
        if self.is_disabled:
            return None
        return self._collected
    
    @property
    def collected_indices(self) -> List[int]:
        """Get list of indices at which values were collected."""
        return self._collected_indices
    
    def get_index_map(self) -> Optional[torch.Tensor]:
        """
        Build dense index map: original_position → compact_index.
        
        Returns a 1D LongTensor of size (total_steps + 1), where entry ``i``
        gives the index into the compact ``all_latents`` for original
        trajectory position ``i``, or -1 if that position was not collected.
        
        When ``collect_all=True``, returns identity ``[0, 1, ..., T]``.
        Cost is negligible (<1KB for typical step counts).
        
        Returns:
            LongTensor of shape (total_steps + 1), or None if collection is disabled.
        
        Example:
            >>> collector = TrajectoryCollector([2, 3, 5, 6], total_steps=8)
            >>> # After collection: collected_indices = [2, 3, 5, 6]
            >>> collector.get_index_map()
            tensor([-1, -1,  0,  1, -1,  2,  3, -1, -1])
        """
        if self.is_disabled:
            return None
        
        total_positions = self.total_steps + 1
        
        if self.collect_all:
            return torch.arange(total_positions, dtype=torch.long)
        
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, original_idx in enumerate(self._collected_indices):
            index_map[original_idx] = compact_idx
        
        return index_map
    
    def reset(self) -> None:
        """Clear collected values for reuse."""
        self._collected = []
        self._collected_indices = []
    
    def __len__(self) -> int:
        """Number of collected values."""
        return len(self._collected)


def compute_trajectory_indices(
    train_timestep_indices: Union[List[int], torch.Tensor],
    num_inference_steps: int,
    include_initial: bool = True,
) -> List[int]:
    """
    Compute the minimal set of trajectory positions needed for training.
    
    For each training timestep index ``i``, the trainer needs positions
    ``i`` (current latents) and ``i + 1`` (next latents). This function
    returns the deduplicated union of all required positions, sorted
    ascending. Consecutive training steps naturally share boundaries,
    further reducing the collected set.
    
    Args:
        train_timestep_indices: Step indices used during training
            (e.g., scheduler.train_timesteps). These are indices into the
            trajectory (0-based), NOT timestep values.
        num_inference_steps: Total denoising steps T.
            The trajectory has T+1 positions (initial + T step outputs).
        include_initial: If True, always include position 0 (initial noise).
            Useful for algorithms that need the starting state.
    
    Returns:
        Sorted list of unique trajectory positions to collect.
    
    Examples:
        >>> compute_trajectory_indices([2, 5, 8], num_inference_steps=20)
        [0, 2, 3, 5, 6, 8, 9]   # 7 positions instead of 21 (3× saving)
        
        >>> compute_trajectory_indices([0, 1, 2], num_inference_steps=20)
        [0, 1, 2, 3]            # Consecutive steps share boundaries
        
        >>> compute_trajectory_indices(range(20), num_inference_steps=20)
        [0, 1, 2, ..., 20]      # Full trajectory (no saving, same as 'all')
    """
    if isinstance(train_timestep_indices, torch.Tensor):
        train_timestep_indices = train_timestep_indices.tolist()
    
    total_positions = num_inference_steps + 1
    positions = set()
    
    if include_initial:
        positions.add(0)
    
    for idx in train_timestep_indices:
        if 0 <= idx < total_positions:
            positions.add(idx)
        if 0 <= idx + 1 < total_positions:
            positions.add(idx + 1)
    
    return sorted(positions)


def create_trajectory_collector(
    indices: TrajectoryIndicesType,
    num_steps: int,
) -> TrajectoryCollector:
    """
    Factory function to create a TrajectoryCollector.
    
    Args:
        indices: Which steps to collect ('all', None, or List[int])
        num_steps: Number of denoising steps
    
    Returns:
        Configured TrajectoryCollector instance
    """
    return TrajectoryCollector(
        indices=indices,
        total_steps=num_steps,
    )