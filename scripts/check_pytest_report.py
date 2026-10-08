"""Verify that no tests are skipped, including expected failures."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def main(path: str) -> int:
    root = ET.parse(Path(path)).getroot()
    skipped = {
        f"{case.attrib['classname']}.{case.attrib['name']}"
        for case in root.iter("testcase")
        if case.find("skipped") is not None
    }
    if skipped:
        print(f"unexpected skipped tests or expected failures: {sorted(skipped)}")
        return 1
    print("verified no skipped tests or expected failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
