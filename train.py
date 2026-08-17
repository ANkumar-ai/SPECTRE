"""Shakespeare, word-level: SPECTRE vs vanilla (learned emb + sinusoidal PE).

Usage (Colab, GPU recommended):
    python train.py --arm both --iters 1500
Outputs: results.json, curves.png
"""
import argparse, json, math, os, re, time, urllib.request
import torch
from model import GPT

URL = ('https://raw.githubusercontent.com/karpathy/char-rnn/'
       'master/data/tinyshakespeare/input.txt')
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z]")


def load_text(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return open(path, encoding='utf-8').read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='both',
                    choices=['both', 'vanilla', 'spectre'])
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--dmodel', type=int, default=128)
    ap.add_argument('--nlayer', type=int, default=4)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--data', default=None, help='path to input.txt')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    text = load_text(args.data)
    words = TOKEN_RE.findall(text)
    vocab = sorted(set(words))
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([stoi[w] for w in words], dtype=torch.long)
    lens = sorted(len(w) for w in vocab)
    pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))]
    print(f'vocab {len(vocab)} | corpus {len(ids)} tokens | '
          f'len p50={pct(.5)} p99={pct(.99)} p999={pct(.999)} max={lens[-1]}')

    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    def get_batch(split):
        data = train_ids if split == 'train' else val_ids
        ix = torch.randint(len(data) - args.block - 1, (args.batch,))
        x = torch.stack([data[i:i + args.block] for i in ix]).to(dev)
        y = torch.stack([data[i + 1:i + args.block + 1] for i in ix]).to(dev)
        return x, y

    @torch.no_grad()
    def eval_loss(model, iters=50):
        model.eval()
        tot = 0.0
        for _ in range(iters):
            x, y = get_batch('val')
            _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / iters

    def run(arm):
        model = GPT(vocab, d_model=args.dmodel, n_layer=args.nlayer,
                    n_head=args.nhead, block_size=args.block,
                    arm=arm, K=args.K).to(dev)
        nparams = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
        print(f'[{arm}] trainable params: {nparams/1e6:.2f}M '
              f'(spectre L_max={getattr(model.tok, "L_max", "-")})')
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.iters, eta_min=args.lr / 10)
        hist = {'iter': [], 'val': []}
        t0 = time.time()
        for it in range(1, args.iters + 1):
            x, y = get_batch('train')
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if it % args.eval_every == 0 or it == args.iters:
                v = eval_loss(model)
                hist['iter'].append(it); hist['val'].append(v)
                print(f'[{arm}] iter {it:5d}  val {v:.4f}  '
                      f'({time.time()-t0:.0f}s)')
        return hist

    arms = ['vanilla', 'spectre'] if args.arm == 'both' else [args.arm]
    results = {a: run(a) for a in arms}
    json.dump(results, open('results.json', 'w'), indent=1)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for a, h in results.items():
            plt.plot(h['iter'], h['val'], marker='o', label=a)
        plt.xlabel('iteration'); plt.ylabel('val loss')
        plt.title('Shakespeare word-level: SPECTRE vs vanilla PE')
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig('curves.png', dpi=150)
        print('wrote results.json, curves.png')
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()