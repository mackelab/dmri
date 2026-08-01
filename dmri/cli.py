import sys

USAGE = """usage: dmri <command> [options]

Commands:
  predict   Run a pretrained model on a standard dMRI folder
  train     Train a model using Hydra configuration
  eval      Run the full research evaluation pipeline

Run `dmri <command> --help` for command-specific help.
"""


def main():
    """Dispatch the public command-line interface."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(USAGE)
        return

    command = sys.argv.pop(1)
    if command == "predict":
        from dmri.predict import main as command_main
    elif command == "train":
        from dmri.train.train_script import main as command_main
    elif command == "eval":
        from dmri.eval.eval_script import main as command_main
    else:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)

    command_main()
