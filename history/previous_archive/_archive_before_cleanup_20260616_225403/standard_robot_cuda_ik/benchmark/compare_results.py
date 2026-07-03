from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import format_markdown_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    args = parser.parse_args()
    rows = []
    for path_str in args.summaries:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
        class Obj:
            pass
        obj = Obj()
        obj.__dict__.update(data)
        rows.append(obj)
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()

