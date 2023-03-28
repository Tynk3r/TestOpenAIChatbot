import tiktoken

class Tokenizer:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)
    
    def tokenIDs_from_list(self, lst):
        tokenIDs = []
        for string in lst:
            variations = [string, string.capitalize(), " " + string, " " + string.capitalize()]
            for variation in variations:
                ids = self.encode(variation)
                for id in ids:
                    tokenIDs += [id]
        tokenIDs = list(dict.fromkeys(tokenIDs))
        return tokenIDs

if __name__ == "__main__":
    tokenizer = Tokenizer()
    while True:
        text = input("Input text to tokenize: ")
        print (tokenizer.tokenIDs_from_list([text, "sorry", "refuse", "inappropriate"]))