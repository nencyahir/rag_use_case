"""python -m rag.cli build [gpo|gep|imac|all] | ask <use_case> "<question>" [--naive]"""
import json
import sys

from . import pipelines


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "build":
        targets = ["gpo", "gep", "imac"] if args[0] == "all" else [args[0]]
        for uc in targets:
            for mode in ("naive", "advanced"):
                if True:
                    print(uc, mode, pipelines.build(uc, mode), "chunks")
    elif cmd == "ask":
        mode = "naive" if "--naive" in args else "advanced"
        print(json.dumps(pipelines.answer(args[0], args[1], mode), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
