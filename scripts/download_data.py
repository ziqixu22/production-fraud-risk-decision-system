from pathlib import Path
import subprocess
import zipfile


def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "ieee-fraud-detection",
        "-p", str(out)
    ], check=True)
    for z in out.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    print(f"IEEE-CIS data available under {out.resolve()}")


if __name__ == "__main__":
    main()
