# src/flow_factory/cli.py
"""
Command-line interface for Flow-Factory.
"""
import sys
import os
import argparse
from pathlib import Path

from .hparams.args import Arguments
from .models.loader import load_model
from .trainers.loader import load_trainer


def train_cli():
    """Train command entry point."""
    parser = argparse.ArgumentParser(
        description="Flow-Factory Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  flow-factory-train --config config/flux_grpo.yaml
  
  # Resume from checkpoint
  flow-factory-train --config config/flux_grpo.yaml --resume logs/run/epoch_10
  
  # Multi-GPU training
  accelerate launch --multi_gpu flow-factory-train --config config/flux_grpo.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = Arguments.load_from_yaml(args.config)
    
    # Load trainer
    print(f"Loading trainer: GRPO")
    trainer = load_trainer(config=config)
    
    trainer.run()


def eval_cli():
    """Evaluation command entry point."""
    parser = argparse.ArgumentParser(description="Flow-Factory Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    
    args = parser.parse_args()
    
    # Load and evaluate
    config = Arguments.load_from_yaml(args.config)
    adapter = load_model(config.model_args, config.training_args)
    
    # Load checkpoint
    adapter.load_checkpoint(args.checkpoint)
    
    print("Evaluation not yet implemented")
    # TODO: Implement evaluation logic


if __name__ == "__main__":
    train_cli()