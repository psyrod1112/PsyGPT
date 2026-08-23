from collections import Counter

def get_word_freqs(text: str) -> dict:
    words = [tuple(word) + ("</w>",) for word in text.split()]
    count_dict = Counter(words)

    return count_dict

def get_pair_freqs(word_freqs: dict) -> dict:
    # 모든 단어를 순회하면서, 그 단어 안의 연속된 두 심볼(symbol) 쌍마다
    # 그 단어의 빈도수만큼 카운트를 더해줌
    # 예: ('l','o','w','</w>')가 5번 나오면, ('l','o') 쌍도 5, ('o','w') 쌍도 5, ('w','</w>') 쌍도 5씩 더해짐
    pair_freqs = Counter()
    for word, count in word_freqs.items():
        pair_list = []
        for i in range(len(word) - 1):
            sub_tuple = (word[i],word[i+1])
            pair_list.append(sub_tuple)
        pair_counter = Counter(pair_list)
        
        for pair, pair_count in pair_counter.items():
            pair_freqs[pair] += pair_count * count
    
    return pair_freqs

def get_most_frequent_pair(pair_freqs: dict) -> tuple:
    max_count = 0
    return_pair = None
    for pair, count in pair_freqs.items():
        if(max_count < count): 
            max_count = count
            return_pair = pair
    
    return return_pair

def merge_pair(pair: tuple, word_freqs: dict) -> dict:
    # word_freqs의 각 단어(튜플)를 순회하면서,
    # pair가 인접해서 나타나는 자리를 찾아 하나의 심볼로 합친 새 튜플을 만들고,
    # 그걸 key로 하는 새 word_freqs를 반환
    new_word_freqs = Counter()
    for word, count in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])  # 두 심볼을 문자열로 합치기
                i += 2   # 두 개를 한번에 소비했으니 2칸 이동
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = count
    return new_word_freqs

def train_bpe(text: str, num_merges: int):
    #일단 단어별로 끊어 (문장 -> 단어)
    #단어를 문자별로 끊어 (단어 -> 문자)
    word_freqs = get_word_freqs(text)
    
    merges = []   # 합쳐진 순서대로 쌓아둘 리스트

    for _ in range(num_merges):
        
        #모든 단어의 문자쌍을 순회하면서 개수 카운트 해
        pair_freq_dict = get_pair_freqs(word_freqs)
        
        #가장 많은 문자 쌍(MostPair)을 뽑아
        freq_pair_tuple = get_most_frequent_pair(pair_freq_dict)
        
        #모든 단어에 대해서 그 MostPair에 대해 문자쌍 합쳐
        word_freqs = merge_pair(freq_pair_tuple, word_freqs)
        
        #MostPair을 merges에 넣어놔 --> 이게 BPE의 규칙이 되는것임
        merges.append(freq_pair_tuple)  # merges 리스트에 이번에 고른 쌍 추가

    return word_freqs, merges

def encode_word(word: str, merges: list) -> tuple:
    #아까 train_bpe는 "주어진 문장"에 대해서 merges를 뽑는거였으면,
    #지금 encode_word는 "새로운 단어"에 대해서 기존 merges을 적용하는 단계임
    symbols = tuple(word) + ("</w>",)   # 처음엔 문자 단위로 쪼갠 상태

    for pair in merges:
        new_tuple = []
        i = 0
        # symbols 안에 이 pair가 인접해서 있으면 합치기
        # (merge_pair에서 한 단어에 대해 했던 것과 똑같은 로직이에요,
        #  다만 이번엔 word_freqs 전체가 아니라 symbols 하나에만 적용)
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                new_tuple.append(symbols[i] + symbols[i+1])  # 두 심볼을 문자열로 합치기
                i += 2   # 두 개를 한번에 소비했으니 2칸 이동
            else:
                new_tuple.append(symbols[i])
                i += 1
        symbols = tuple(new_tuple)
    return symbols

def build_vocab(word_freqs_initial: dict, merges: list) -> dict:
    symbols = set()
    for word in word_freqs_initial:
        symbols.update(word)   # 아직 merge 안 한 초기 상태 -> 기본 문자 + </w> 다 뽑기
    for a, b in merges:
        symbols.add(a + b)        # 이 merge가 만들어내는 새 심볼 추가
    sorted_symbols = sorted(symbols)
    return {sym: idx for idx, sym in enumerate(sorted_symbols)}


class BPETokenizer:
    def __init__(self, text:str = None, num_merges: int = None,
                 vocab: dict = None, merges: list = None):
        
        if vocab is not None and merges is not None:
            self.vocab = vocab
            self.merges = merges
        else:
            word_freqs_initial = get_word_freqs(text)
            _, self.merges = train_bpe(text, num_merges)
            # 규칙 때문에 필요한 self.merges
            self.vocab = build_vocab(word_freqs_initial, self.merges)
            #id 떄문에 필요한 self.vocab
        
    def encode(self, text:str) -> list[int]:
        ids = []
        for word in text.split():
            symbols = encode_word(word, self.merges)  # encode_word 재사용
            for s in symbols:
                ids.append(self.vocab[s])  # vocab으로 정수 변환해서 추가
        return ids

    def decode(self, ids: list[int]) -> str:
        id_to_symbol = {idx:sym for sym, idx in self.vocab.items()}   # vocab을 뒤집어서 정수->심볼 매핑 만들기 (CharTokenizer 때 했던 것과 같은 패턴)
        symbols = [id_to_symbol[id] for id in ids]      # ids 하나하나를 심볼로 변환
        text = "".join(symbols)
        text = text.replace("</w>", " ")            # "</w>"를 공백(" ")으로 교체 (문자열 .replace() 메서드)
        return text.strip()   # 양 끝 공백 정리 (마지막 단어의 </w>가 남긴 공백 제거)
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"

    tok = BPETokenizer(text, 10)
    ids = tok.encode("wider newest")
    print(tok.decode(ids))