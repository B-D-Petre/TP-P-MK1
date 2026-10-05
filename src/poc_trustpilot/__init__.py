import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="poc-trustpilot", description="Trustpilot interest-extraction demo")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the pipeline: bronze -> silver -> LLM extraction -> graph -> gold")
    run_p.add_argument("--sample", type=int, default=None, help="number of reviews to send to the LLM (default 50)")
    args = parser.parse_args()

    if args.command == "run":
        from .Pipeline.config import SAMPLE_SIZE
        from .Pipeline.run import run

        run(args.sample or SAMPLE_SIZE)
