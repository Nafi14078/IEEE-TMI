"""Cross-platform download of ISIC 2019 into data/raw/isic2019 (works on Windows without wget/unzip).
PAD-UFES-20 must still be downloaded by hand (see message at the end).

python scripts/download_data.py
"""
import os
import urllib.request
import zipfile

BASE = "https://isic-challenge-data.s3.amazonaws.com/2019/"
FILES = ["ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input.zip"]
OUT = os.path.join("data", "raw", "isic2019")


def download(url, dst):
    if os.path.exists(dst):
        print("exists:", dst)
        return
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r{os.path.basename(dst)}: {got / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    os.replace(tmp, dst)
    print()


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join("data", "raw", "pad_ufes20", "images"), exist_ok=True)
    for f in FILES:
        download(BASE + f, os.path.join(OUT, f))
    if not os.path.isdir(os.path.join(OUT, "ISIC_2019_Training_Input")):
        print("unzipping ...")
        with zipfile.ZipFile(os.path.join(OUT, "ISIC_2019_Training_Input.zip")) as z:
            z.extractall(OUT)
    print("\nISIC 2019 ready. Now download PAD-UFES-20 (CC BY 4.0) by hand from\n"
          "  https://data.mendeley.com/datasets/zr7vgbcyr2/1\n"
          "put metadata.csv in data/raw/pad_ufes20/ and unzip all images into data/raw/pad_ufes20/images/")


if __name__ == "__main__":
    main()
