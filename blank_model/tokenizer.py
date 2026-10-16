"""A deliberately simple byte tokenizer for the first training milestone."""

BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 259


class ByteTokenizer:
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(int(i) for i in ids if 0 <= int(i) <= 255)
        return raw.decode("utf-8", errors="replace")
