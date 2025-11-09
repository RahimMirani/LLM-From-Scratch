import re

# LLM Tokenizer

# 1. Creating Tokens
with open("the-verdict.txt", encoding="utf-8") as f:
    raw_text = f.read()

print("Total characters:", len(raw_text))
print(raw_text[:100])

#This code uses re library to split the text first, and takes all the words and punctuations as a seperate token.
# It then removes the white spaces and then keeps the words seperate as tokens

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:100])


# 2. Converting Tokens into Token IDs
all_unique_words = sorted(set(preprocessed))
vocab_size = len(all_unique_words)

print(vocab_size)

#Assigning token ids to words

vocab = {token:integer for integer, token in enumerate(all_unique_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i>=20:
        break 


#Simple Tokenizer Class for encoding and decoding with encoding and decoding methods
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove extra spaces before punctuation
        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text
    

# We create a instance of SimpleTokenizerV1 class, pass the vocab that that contains words and their ids maping that we created above.
# Then we just use the encode method and create the ids of the new text passed 
# The text that is passed is in the training set and in the vocab

tokenizer = SimpleTokenizerV1(vocab)

text = """ "It's the last he painted, you know,"
            Mr's Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

# Decoding the text by using the decode method and passing the same ids generated above

decoded_text = tokenizer.decode(ids)
print(decoded_text)

# Passing two new token, ENDOFTEXT AND UNK to the vocab
# So that the encoder does not throw error if it sees unk words

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab1 = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab1))
print(vocab1)


# Same class but with handling the UNK words
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        #replaces the word with unk id if not present in the vocab training data
        preprocessed =[
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove extra spaces before punctuation
        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text
    

# Passed the updated vocab1 with unk int mapping
tokenizer = SimpleTokenizerV2(vocab1)

text1 = "Hello, do you like tea or not?"
text2 = "In the sinlight terraces of the palance in the woods"

# Combined both the text and added endoftext in between
text = " <|endoftext|> ".join((text1, text2))

print(text)

exmp = tokenizer.encode(text)
print(exmp)

tokenizer.decode(exmp)