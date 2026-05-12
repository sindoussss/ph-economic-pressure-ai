import requests
from tqdm import tqdm
import os

# Simple English Wikipedia — much smaller (~250MB), more carefully reviewed
# than full English Wikipedia. Better for RAG: concise, cleaner articles.
# Full English Wikipedia is ~22GB compressed and has more unverified content.
url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
filename = os.path.join(os.path.dirname(__file__), "..", "simplewiki-latest-pages-articles.xml.bz2")


# First, install tqdm if you don't have it
# Run: pip install tqdm

def download_with_progress(url, filename):
    print(f"Downloading: {filename}")
    print(f"From: {url}")

    # Get file size for progress bar
    response = requests.head(url)
    total_size = int(response.headers.get('content-length', 0))

    print(f"Total size: {total_size / (1024 ** 3):.2f} GB")

    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for errors

    # Initialize progress bar
    with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

    print("✅ Download complete!")


if __name__ == "__main__":
    download_with_progress(url, filename)