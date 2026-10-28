"""Sample questions from RAG-Hands-On-Use-Cases, each with a string that the right passage must contain."""
from . import pipelines

CASES = {
    "gpo": [
        ("How many depository libraries did GPO staff visit in FY 2018?", "librar"),
        ("What new digital system replaced FDsys, and when was it fully deployed?", "govinfo"),
        ("How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?", "procure"),
    ],
    "gep": [
        ("What is the growth outlook for South Asia in this report?", "South Asia"),
        ("What does Chapter 3 say about sovereign debt in developing economies?", "debt"),
        ("According to the executive summary, what are the main risks to the global outlook?", "risk"),
    ],
    "imac": [
        ("How much less energy does iMac use compared to the ENERGY STAR requirement?", "58 percent"),
        ("What is the total carbon footprint of the iMac, and how much comes from product use versus manufacturing?", "Product use: 45%"),
        ("What material is the iMac stand made from, and what percentage is recycled?", "100 percent recycled aluminum"),
    ],
}


def run(generate: bool = False, mode: str = "advanced", use_llm: bool = False):
    for uc, cases in CASES.items():
        for q, needle in cases:
            hits, trace = pipelines.retrieve(uc, q, mode, use_llm)
            rank = next((i for i, h in enumerate(hits, 1) if needle.lower() in h["text"].lower()), None)
            print(f"[{uc}] rank={rank}  {q}")
            if generate:
                print("   ->", pipelines.answer(uc, q, mode, use_llm)["answer"])


if __name__ == "__main__":
    import sys
    run(generate="--generate" in sys.argv, mode="naive" if "--naive" in sys.argv else "advanced", use_llm="--llm" in sys.argv)
