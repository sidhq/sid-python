"""Verify that only the documented strict xfail is skipped."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


EXPECTED_XFAILS = {
    "tests.test_sdk_invariants.test_fully_seen_row_drops_display_attrs",
}


def main(path: str) -> int:
    root = ET.parse(Path(path)).getroot()
    skipped = {
        f"{case.attrib['classname']}.{case.attrib['name']}"
        for case in root.iter("testcase")
        if case.find("skipped") is not None
    }
    if skipped != EXPECTED_XFAILS:
        print(f"expected xfails {sorted(EXPECTED_XFAILS)}, got {sorted(skipped)}")
        return 1
    print("verified the documented strict xfail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
