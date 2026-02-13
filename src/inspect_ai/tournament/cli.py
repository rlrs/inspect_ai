import argparse
from typing import Sequence

from ._cli_format import (
    export_result_payload,
    format_export_result,
    format_generation_result,
    format_run_result,
    format_status,
    generation_result_payload,
    run_result_payload,
    status_payload,
    write_json_output,
)
from .exports import export_rankings
from .generation import run_generation
from .orchestrator import (
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
            payload = generation_result_payload(generation_result)
            output_path = write_json_output(payload, args.json_out)
            print(format_generation_result(generation_result))
            if output_path is not None:
                print(f"\nSaved JSON output to {output_path.as_posix()}")
            return 0

        if args.command == "run":
            run_result = run_tournament(args.config, max_batches=args.max_batches)
            payload = run_result_payload(run_result)
            output_path = write_json_output(payload, args.json_out)
            print(format_run_result(run_result))
            if output_path is not None:
                print(f"\nSaved JSON output to {output_path.as_posix()}")
            return 0

        if args.command == "resume":
            resume_result = resume_tournament(args.target, max_batches=args.max_batches)
            payload = run_result_payload(resume_result)
            output_path = write_json_output(payload, args.json_out)
            print(format_run_result(resume_result))
            if output_path is not None:
                print(f"\nSaved JSON output to {output_path.as_posix()}")
            return 0

        if args.command == "status":
            status = tournament_status(args.target)
            payload = status_payload(status)
            output_path = write_json_output(payload, args.json_out)
            print(format_status(status))
            if output_path is not None:
                print(f"\nSaved JSON output to {output_path.as_posix()}")
            return 0

        if args.command == "export":
            export_result = export_rankings(args.target, output_dir=args.output_dir)
            payload = export_result_payload(export_result)
            output_path = write_json_output(payload, args.json_out)
            print(format_export_result(export_result))
            if output_path is not None:
                print(f"\nSaved JSON output to {output_path.as_posix()}")
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
    generate_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    run_parser = subparsers.add_parser("run", help="Run tournament from config")
    run_parser.add_argument("--config", required=True, help="Path to tournament config")
    run_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    run_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
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
    resume_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    status_parser = subparsers.add_parser("status", help="Show tournament status")
    status_parser.add_argument("target", help="Config file path or state dir path")
    status_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    export_parser = subparsers.add_parser("export", help="Export rankings artifacts")
    export_parser.add_argument("target", help="Config file path or state dir path")
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for export files",
    )
    export_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
