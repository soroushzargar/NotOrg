"""CLI entry point for NotOrg.

Usage examples::

    notorg --store "I want to use conformal prediction on LLM graph structures"
    notorg --organize
    notorg --fetch "any ideas about conformal prediction?"
    notorg --list-pending
"""

import argparse
import json
import sys
import textwrap

from notorg import store, organize, fetch, storage


def _print_results(results: list[dict]) -> None:
    """Pretty-print fetch results."""
    if not results:
        print("No matching ideas found.")
        return
    print(f"\nFound {len(results)} matching idea(s):\n")
    for item in results:
        path_str = " -> ".join(item["path"]) if item["path"] else "(root)"
        idea = item["idea"]
        print(f"  [{path_str}]")
        wrapped = textwrap.fill(idea["text"], width=72, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)
        print(f"    (stored: {idea.get('timestamp', 'unknown')})")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="notorg",
        description="NotOrg – store, organise, and retrieve your ideas.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--store",
        metavar="IDEA",
        help="Store a new idea in the pending queue.",
    )
    group.add_argument(
        "--organize",
        action="store_true",
        help="Organise all pending ideas into the mind-map using the LLM.",
    )
    group.add_argument(
        "--fetch",
        metavar="QUERY",
        help="Search the mind-map for ideas related to QUERY.",
    )
    group.add_argument(
        "--list-pending",
        action="store_true",
        help="Show all pending (unorganised) ideas.",
    )
    group.add_argument(
        "--show-mindmap",
        action="store_true",
        help="Print the raw mind-map JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args(argv)

    try:
        if args.store is not None:
            entry = store.store_idea(args.store)
            print(f"Stored idea #{entry['id']}: \"{entry['text']}\"")

        elif args.organize:
            organize.organize_ideas(verbose=True)

        elif args.fetch is not None:
            results = fetch.fetch_ideas(args.fetch, verbose=args.verbose)
            _print_results(results)

        elif args.list_pending:
            pending = storage.load_pending()
            if not pending:
                print("No pending ideas.")
            else:
                print(f"{len(pending)} pending idea(s):")
                for idea in pending:
                    print(f"  #{idea['id']} [{idea.get('timestamp', '')}] {idea['text']}")

        elif args.show_mindmap:
            tree = storage.load_mindmap()
            print(json.dumps(tree, indent=2))

    except (ValueError, KeyboardInterrupt) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
