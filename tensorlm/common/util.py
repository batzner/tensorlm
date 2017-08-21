def get_chunks(items, size):
    """Get successive chunks from items."""
    chunks = []
    for i in range(0, len(items), size):
        chunks.append(items[i:i + size])
    return chunks
