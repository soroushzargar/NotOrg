"""--store command: append an idea to the pending file."""

from notorg import storage


def store_idea(text: str) -> dict:
    """Append *text* as a new pending idea and return the stored entry."""
    text = text.strip()
    if not text:
        raise ValueError("Idea text must not be empty.")
    entry = storage.append_pending(text)
    return entry
