"""
Project Gutenberg downloader for offline RAG.

Downloads free public-domain books (literature, history, science, philosophy).
All texts are vetted — these are published books, not user-edited articles.
Much more reliable as a knowledge source than Wikipedia for factual/literary content.

Install deps:  pip install tqdm requests
"""
import os
import requests
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "gutenberg_books")

# Curated list: book ID → title
# Find more at https://www.gutenberg.org/browse/scores/top
BOOKS = {
    1342: "Pride and Prejudice",
    11:   "Alice's Adventures in Wonderland",
    1661: "The Adventures of Sherlock Holmes",
    2701: "Moby Dick",
    84:   "Frankenstein",
    98:   "A Tale of Two Cities",
    1080: "A Modest Proposal",
    46:   "A Christmas Carol",
    74:   "The Adventures of Tom Sawyer",
    16328: "Beowulf",
    5740: "Thus Spoke Zarathustra",
    1232: "The Prince (Machiavelli)",
    2542: "A Doll's House",
    3207: "The Republic (Plato)",
    17192: "Philippine Folk Tales",         # relevant for Maria's Filipino context
    6925:  "Noli Me Tangere (English)",     # directly relevant to Project Maria
}


def download_book(book_id: int, title: str, output_dir: str):
    """Download a single Project Gutenberg book as plain text."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{book_id}_{title.replace(' ', '_')}.txt")

    if os.path.exists(filename):
        print(f"  ⏭ Already downloaded: {title}")
        return

    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    fallback_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    for attempt_url in (url, fallback_url):
        try:
            response = requests.get(attempt_url, timeout=30)
            if response.status_code == 200:
                with open(filename, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(response.text)
                print(f"  ✅ {title} ({len(response.text) // 1024} KB)")
                return
        except requests.RequestException:
            continue

    print(f"  ❌ Failed: {title} (ID {book_id})")


def download_all(books: dict = BOOKS, output_dir: str = OUTPUT_DIR):
    print(f"Downloading {len(books)} Project Gutenberg books to: {output_dir}\n")
    for book_id, title in tqdm(books.items(), desc="Books", unit="book"):
        download_book(book_id, title, output_dir)
    print(f"\nDone. Books saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    download_all()
