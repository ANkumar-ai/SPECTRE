import random
import re

def typo_word(w):
    if len(w) < 3: return w
    op = random.choice(['swap', 'delete', 'sub'])
    if op == 'swap' and len(w) >= 2:
        i = random.randrange(len(w)-1)
        w = w[:i] + w[i+1] + w[i] + w[i+2:]
    elif op == 'delete':
        i = random.randrange(len(w))
        w = w[:i] + w[i+1:]
    elif op == 'sub':
        i = random.randrange(len(w))
        new_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        while new_char == w[i].lower():
            new_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        w = w[:i] + new_char + w[i+1:]
    return w

def inject_typos(text, word_fraction=0.1):
    parts = re.split(r'(\s+)', text)
    for i, part in enumerate(parts):
        if re.match(r'[A-Za-z]+', part):
            if random.random() < word_fraction:
                parts[i] = typo_word(part)
    return ''.join(parts)

if __name__ == '__main__':
    import sys
    in_path = sys.argv[1] if len(sys.argv)>1 else 'data/input.txt'
    out_path = sys.argv[2] if len(sys.argv)>2 else 'data/input_typo.txt'
    with open(in_path, 'r', encoding='utf-8') as f:
        text = f.read()
    typo_text = inject_typos(text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(typo_text)
    print(f'Generated {out_path}')