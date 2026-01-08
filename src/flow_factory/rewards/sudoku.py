# src/flow_factory/rewards/sudoku.py
from accelerate import Accelerator
from typing import Optional, List, Union
from PIL import Image
import torch
import copy
import numpy as np

from transformers import AutoProcessor, AutoModelForImageTextToText

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *


class SudokuRewardModel(BaseRewardModel):
    def __init__(self, reward_args: RewardArguments, accelerator: Accelerator):
        super().__init__(reward_args, accelerator)
        self.size = 9
        self.img_size = 512
        self.cell_size = self.img_size / self.size
        self.model = AutoModelForImageTextToText.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf", device_map=self.device, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

    def _to_pil(self, img: Union[Image.Image, torch.Tensor, np.ndarray, List]) -> List[Image.Image]:
        """Convert tensor/ndarray/PIL to list of PIL Images."""
        if isinstance(img, list):
            return sum([self._to_pil(x) for x in img], [])
        if isinstance(img, Image.Image):
            return [img]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[None, ..., None]
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))
            img = img[None]
        elif img.ndim == 4 and img.shape[1] in (1, 3, 4) and img.shape[1] < img.shape[-1]:
            img = np.transpose(img, (0, 2, 3, 1))
        vmin, vmax = img.min(), img.max()
        if vmin >= -1.0 and vmax <= 1.0 and vmin < 0:
            img = (img + 1) * 127.5
        elif vmax <= 1.0:
            img = img * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return [Image.fromarray(x.squeeze(-1) if x.shape[-1] == 1 else x) for x in img]

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        condition_images = [self._to_pil(cond_imgs) for cond_imgs in condition_images]
        batch_size = len(prompt)
        
        # Collect all images for batch parsing
        puzzles = [cond[0] for cond in condition_images]
        solutions = list(image)
        
        # Batch parse all images
        all_grids = self._parse_grids_batch(puzzles + solutions)
        puzzle_grids = all_grids[:batch_size]
        solution_grids = all_grids[batch_size:]
        
        # Compute rewards
        rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            _, _, reward = self._compute_single_reward(puzzle_grids[i], solution_grids[i])
            rewards[i] = reward
        
        return RewardModelOutput(rewards=rewards, extra_info={})
    
    def _crop_cells(self, img: Image.Image) -> List[Image.Image]:
        """Crop 81 cells from a sudoku image."""
        img = img.convert('RGB').resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        cs = self.cell_size
        cells = []
        for idx in range(81):
            row, col = divmod(idx, 9)
            x1, y1 = int(col * cs) + 2, int(row * cs) + 2
            x2, y2 = int((col + 1) * cs) - 2, int((row + 1) * cs) - 2
            cells.append(img.crop((x1, y1, x2, y2)))
        return cells

    @torch.no_grad()
    def _batch_ocr(self, images: List[Image.Image]) -> List[str]:
        """Batch OCR using GOT-OCR-2.0 with batch size limit."""
        results = []
        for i in range(0, len(images), self.reward_args.batch_size):
            batch = images[i : i + self.reward_args.batch_size]
            inputs = self.processor(batch, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=1,
            )
            texts = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            results.extend(texts)
        return results

    def _parse_grids_batch(self, images: List[Image.Image]) -> List[List[List[int]]]:
        """Parse multiple sudoku images to grids using batch OCR."""
        all_cells = []
        for img in images:
            all_cells.extend(self._crop_cells(img))
        
        # Batch OCR all cells
        ocr_results = self._batch_ocr(all_cells)
        
        # Reconstruct grids
        grids = []
        for img_idx in range(len(images)):
            grid = [[0] * 9 for _ in range(9)]
            for cell_idx in range(81):
                text = ocr_results[img_idx * 81 + cell_idx]
                row, col = divmod(cell_idx, 9)
                for ch in text:
                    if ch.isdigit() and ch != '0':
                        grid[row][col] = int(ch)
                        break
            grids.append(grid)
        return grids

    def _compute_single_reward(self, puzzle: List[List[int]], solution: List[List[int]]) -> tuple:
        """Compute reward: a1 (new digit accuracy) - a2 (old digit modification rate)."""
        gt_solutions = self._find_solutions(puzzle, limit=1)
        gt = gt_solutions[0] if gt_solutions else None
        
        new_correct = new_total = old_modified = old_total = 0
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    new_total += 1
                    if gt and solution[i][j] == gt[i][j]:
                        new_correct += 1
                else:
                    old_total += 1
                    if solution[i][j] != puzzle[i][j]:
                        old_modified += 1
        
        a1 = new_correct / new_total if new_total else 1.0
        a2 = old_modified / old_total if old_total else 0.0
        return a1, a2, a1 - a2

    def _find_solutions(self, puzzle: List[List[int]], limit: int = 1) -> List[List[List[int]]]:
        """Backtracking solver."""
        solutions, grid = [], copy.deepcopy(puzzle)
        
        def is_valid(r, c, num):
            if num in grid[r]: return False
            if num in [grid[i][c] for i in range(9)]: return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            return all(grid[br+i][bc+j] != num for i in range(3) for j in range(3))
        
        def backtrack():
            if len(solutions) >= limit: return
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        for num in range(1, 10):
                            if is_valid(i, j, num):
                                grid[i][j] = num
                                backtrack()
                                grid[i][j] = 0
                        return
            solutions.append(copy.deepcopy(grid))
        
        backtrack()
        return solutions