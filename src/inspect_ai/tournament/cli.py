import argparse
import json
from typing import Sequence

from .exports import export_rankings
from .generation import run_generation
from .orchestrator import (
    TournamentRunResult,
    resume_tournament,
    run_tournament,
    tournament_status,
)
from .store import initialize_tournament_store


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "init":
            db_path = initialize_tournament_store(args.config)
            print(db_path)
            return 0

        if args.command == "generate":
            models = args.models if args.models else None
            generation_result = run_generation(args.config, models=models)
            print(
                json.dumps(
                    {
                        "models": generation_result.models,
                        "prompt_count": generation_result.prompt_count,
                        "log_dir": generation_result.log_dir.as_posix(),
                        "log_count": generation_result.log_count,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "run":
            run_result = run_tournament(args.config, max_batches=args.max_batches)
            _print_run_result(run_result)
            return 0

        if args.command == "resume":
            resume_result = resume_tournament(args.target, max_batches=args.max_batches)
            _print_run_result(resume_result)
            return 0

        if args.command == "status":
            status = tournament_status(args.target)
            print(status.model_dump_json(indent=2))
            return 0

        if args.command == "export":
            export_result = export_rankings(args.target, output_dir=args.output_dir)
            print(
                json.dumps(
                    {
                        "output_dir": export_result.output_dir.as_posix(),
                        "rankings_json": export_result.rankings_json.as_posix(),
                        "rankings_csv": export_result.rankings_csv.as_posix(),
                        "pairwise_matrix_csv": (
                            export_result.pairwise_matrix_csv.as_posix()
                            if export_result.pairwise_matrix_csv is not None
                            else None
                        ),
                    },
                    indent=2,
                )
            )
            return 0

        parser.print_help()
        return 1
    except Exception as ex:
        print(f"ERROR: {ex}")
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspect_ai.tournament.cli")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize tournament state")
    init_parser.add_argument("--config", required=True, help="Path to tournament config")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate contestant responses"
    )
    generate_parser.add_argument(
        "--config", required=True, help="Path to tournament config"
    )
    generate_parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of contestant models",
    )

    run_parser = subparsers.add_parser("run", help="Run tournament from config")
    run_parser.add_argument("--config", required=True, help="Path to tournament config")
    run_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )

    resume_parser = subparsers.add_parser(
        "resume", help="Resume tournament from config path or state dir"
    )
    resume_parser.add_argument("target", help="Config file path or state dir path")
    resume_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )

    status_parser = subparsers.add_parser("status", help="Show tournament status")
    status_parser.add_argument("target", help="Config file path or state dir path")

    export_parser = subparsers.add_parser("export", help="Export rankings artifacts")
    export_parser.add_argument("target", help="Config file path or state dir path")
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for export files",
    )

    return parser


def _print_run_result(result: TournamentRunResult) -> None:
    print(
        json.dumps(
            {
                "batches_completed": result.batches_completed,
                "matches_scheduled": result.matches_scheduled,
                "outcomes_processed": result.outcomes_processed,
                "outcomes_skipped": result.outcomes_skipped,
                "status": result.status.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
