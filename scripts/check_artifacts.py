"""Check release contents, metadata, licenses, and embedded Rust SBOMs."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import zipfile


FORBIDDEN_REQUIREMENTS = ("numpy", "scipy", "scikit-learn", "sklearn")


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        required_suffixes = (
            "sid/_snippet.pyi",
            "sid/py.typed",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/THIRD_PARTY_NOTICES.md",
        )
        for suffix in required_suffixes:
            assert any(name.endswith(suffix) for name in names), (path, suffix)
        assert any(".dist-info/sboms/" in name and name.endswith(".json") for name in names)
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata["Name"] == "sid-sdk"
        assert metadata["Version"] == "0.2.0"
        assert metadata["License-Expression"] == "MIT"
        requirements = "\n".join(metadata.get_all("Requires-Dist", []))
        assert not any(name in requirements.lower() for name in FORBIDDEN_REQUIREMENTS)


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    for suffix in (
        "/Cargo.lock",
        "/Cargo.toml",
        "/LICENSE",
        "/README.md",
        "/THIRD_PARTY_NOTICES.md",
        "/rust-toolchain.toml",
        "/sid/_snippet.pyi",
        "/sid/py.typed",
        "/src/lib.rs",
    ):
        assert any(name.endswith(suffix) for name in names), (path, suffix)
    assert not any("/dist/" in name for name in names)
    assert not any(Path(name).name.startswith(".env") for name in names)


def main(directory: str) -> None:
    paths = list(Path(directory).iterdir())
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    assert wheels, "no wheels found"
    assert len(sdists) == 1, f"expected one sdist, found {len(sdists)}"
    for wheel in wheels:
        check_wheel(wheel)
    check_sdist(sdists[0])
    print(f"validated {len(wheels)} wheel(s) and one sdist")


if __name__ == "__main__":
    main(sys.argv[1])
