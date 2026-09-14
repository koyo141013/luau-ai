import sys

sys.path.insert(0, "src")

from tokenizer import LuauTokenizer


tokenizer = LuauTokenizer()

code = 'local Players = game:GetService("Players")'

tokenizer.build_vocab([code])

print("=== TinyLuauGPT Tokenizer Test ===")
print()
print("Original:")
print(code)
print()
print("Tokens:")
print(tokenizer.tokenize(code))
print()
print("Vocab size:")
print(tokenizer.vocab_size)
print()

ids = tokenizer.encode(code)

print("Token IDs:")
print(ids)
print()

print("Decoded:")
print(tokenizer.decode(ids))
print()

if tokenizer.decode(ids) == code:
    print("SUCCESS! Tokenizer is working!")
else:
    print("WARNING: Decoded text is different.")