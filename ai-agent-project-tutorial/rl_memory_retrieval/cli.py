"""CLI interface for the RL Memory Retrieval pipeline.

Usage:
    rl-retriever train --source <url|file|dir> --output ./model_output
    rl-retriever query --model ./model_output --question "What is X?"
"""

from __future__ import annotations

import logging
import sys

import click

from rl_memory_retrieval.config import Settings
from rl_memory_retrieval.pipeline import Pipeline


@click.group()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
def cli(verbose: bool) -> None:
    """RL Memory Retrieval — Train and query RL-powered memory retrieval."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
@click.option(
    "--source",
    required=True,
    help="Knowledge source: URL, file path, or directory.",
)
@click.option(
    "--output",
    default="./model_output",
    show_default=True,
    help="Output directory for trained model artifacts.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to a custom config.yaml file.",
)
def train(source: str, output: str, config_path: str | None) -> None:
    """Train a new RL retriever from a knowledge source."""
    try:
        if config_path:
            pipeline = Pipeline.from_config(config_path)
        else:
            pipeline = Pipeline(Settings(source=source))

        result = pipeline.train(source=source, output_dir=output)

        click.echo("\n" + "=" * 50)
        click.echo("Training Complete!")
        click.echo("=" * 50)
        click.echo(f"  Chunks:  {result['num_chunks']}")
        click.echo(f"  Queries: {result['num_queries']}")
        click.echo(f"  Train:   {result['splits']['train']}")
        click.echo(f"  Val:     {result['splits']['val']}")
        click.echo(f"  Test:    {result['splits']['test']}")

        eval_results = result.get("evaluation", {})
        if "baseline" in eval_results:
            click.echo(
                f"  Baseline Accuracy: "
                f"{eval_results['baseline']['accuracy']:.1%}"
            )
        if "rl" in eval_results:
            click.echo(
                f"  RL Accuracy:       "
                f"{eval_results['rl']['accuracy']:.1%}"
            )
        click.echo(f"\n  Model saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--model",
    required=True,
    help="Path to trained model directory.",
)
@click.option(
    "--question",
    required=True,
    help="Question to answer.",
)
def query(model: str, question: str) -> None:
    """Query a trained RL retriever."""
    try:
        pipeline = Pipeline(Settings())
        result = pipeline.query(question=question, model_dir=model)

        click.echo(f"\nQuestion: {question}")
        click.echo(f"Answer:   {result['answer']}")
        click.echo(f"Method:   {result['method']}")
        click.echo(f"Sim:      {result['sim']:.4f}")
        click.echo(f"Memory:   #{result['memory_id']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
