class CharTokenizer:
    def __init__(self, text: str):
        token_key = set(text)
        token_key = sorted(token_key)
        self.token_list = dict()
        self.token_ids = dict()
        for idx, key in enumerate(token_key):
            self.token_list[idx] = key
            self.token_ids[key] = idx
        

    def encode(self, s: str) -> list[int]:
        return [self.token_ids[word] for word in s]

    def decode(self, ids: list[int]) -> str:
        decode_list = [self.token_list[id] for id in ids]
        return "".join(decode_list)

    @property
    def vocab_size(self) -> int:
        return len(self.token_list)

tok = CharTokenizer("hello world")
ids = tok.encode("hello")
print(ids)
print(tok.decode(ids))  # "hello"가 나와야 함
assert tok.decode(tok.encode("hello world")) == "hello world"
