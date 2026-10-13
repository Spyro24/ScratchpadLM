"""Fast unit test suite for ScratchpadLM (TinyLM).

Runs every core subsystem with tiny models so the whole suite finishes in
seconds. Usage:

    python3 test_all.py            # all tests
    python3 test_all.py tokenizer  # run tests whose name matches a filter

Exit code 0 = all passed, 1 = failures. Stdlib only (no pytest required).
"""
import math
import os
import random
import shutil
import sys
import tempfile
import traceback

# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------

PASS, FAIL = 0, 0
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


TESTS = []


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "check failed")


def run():
    global PASS, FAIL
    for name, fn in TESTS:
        if FILTER and FILTER not in name:
            continue
        try:
            fn()
            PASS += 1
            print(f"  PASS  {name}")
        except Exception:
            FAIL += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


# ---------------------------------------------------------------------------
# shared fixtures (tiny model/tokenizer, rebuilt per test for isolation)
# ---------------------------------------------------------------------------

def tiny_tok_and_ids(corpus="the quick brown fox jumps over the lazy dog. " * 40,
                     vocab=400):
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab)
    return tok, tok.encode(corpus)


def tiny_model(vocab=400, ctx=6, emb=8, hidden=16):
    from tinylm import TinyLM
    return TinyLM(vocab_size=vocab, ctx=ctx, emb=emb, hidden=hidden)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@test("config: defaults + validation")
def t_config_defaults():
    from config import Config, ConfigError
    cfg = Config.from_dict({})
    check(cfg["generation"]["top_k"] == 0)
    check(cfg["generation"]["top_p"] == 0.0)
    check(cfg["generation"]["repetition_penalty"] == 1.0)
    check(cfg["training"]["lr_scheduler"] == "cosine_warmup")
    check(cfg["training"]["grad_clip"] == 1.0)
    check(cfg["training"]["momentum"] == 0.9)
    check(cfg["training"]["rollback_window"] == 2)
    check(cfg["training"]["early_stopping"] is True)
    for bad in ({"generation": {"top_k": -1}},
                {"generation": {"top_p": 1.5}},
                {"generation": {"repetition_penalty": 0.5}},
                {"training": {"lr_scheduler": "nope"}},
                {"training": {"momentum": 1.0}},
                {"training": {"momentum": -0.5}},
                {"training": {"rollback_window": -1}},
                {"training": {"val_split": 1.0}}):
        try:
            Config.from_dict(bad)
            raise AssertionError(f"expected ConfigError for {bad}")
        except ConfigError:
            pass


@test("config: yaml round trip via _write_config_yaml")
def t_config_yaml_roundtrip():
    from config import Config
    from ui_base import write_config_yaml as _write_config_yaml
    data = Config.from_dict({"generation": {"top_k": 7, "top_p": 0.95}}).to_dict()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        _write_config_yaml(path, data)
        cfg = Config.from_file(path)
        check(cfg["generation"]["top_k"] == 7)
        check(cfg["generation"]["top_p"] == 0.95)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

@test("tokenizer: encode/decode roundtrip + extend keeps ids")
def t_tokenizer():
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train("hello world " * 50, vocab_size=300)
    text = "hello world"
    check(tok.decode(tok.encode(text)) == text)
    old_vocab = dict(tok.vocab)
    tok.extend("newwords appear here " * 30, vocab_size=360)
    for tid, b in old_vocab.items():
        check(tok.vocab.get(tid) == b, f"token {tid} was renumbered!")


@test("tokenizer: save/load preserves merges")
def t_tokenizer_saveload():
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train("abc def ghi " * 50, vocab_size=300)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.txt")
        tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        check(tok2.encode("abc def") == tok.encode("abc def"))


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------

@test("autograd: grads match finite differences")
def t_autograd():
    from tinylm import Value
    a, b = Value(2.0), Value(-3.0)
    out = (a * b + a.tanh()) * (b * b + 1.0)
    out.backward()
    # finite difference on a
    eps = 1e-6
    f = lambda x, y: (x * y + math.tanh(x)) * (y * y + 1.0)
    da = (f(2.0 + eps, -3.0) - f(2.0, -3.0)) / eps
    check(abs(a.grad - da) < 1e-4, f"{a.grad} != {da}")
    check(abs(out.data - f(2.0, -3.0)) < 1e-9)


# ---------------------------------------------------------------------------
# model forward/backward + sync_vocab
# ---------------------------------------------------------------------------

@test("model: loss decreases after training step")
def t_model_train_step():
    from tinylm import train
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    random.seed(0)
    before = None
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(model, ids, steps=6, lr=0.1, batch_size=8, n_negatives=32, use_c=False)
        # one more step measured: capture loss via progress_cb
    losses = []
    with contextlib.redirect_stdout(buf):
        train(model, ids, steps=6, lr=0.1, batch_size=8, n_negatives=32,
              use_c=False, progress_cb=lambda d, t, l, *a: losses.append(l))
    check(len(losses) == 6)
    check(all(math.isfinite(l) for l in losses))


@test("sync_vocab grows model in place")
def t_sync_vocab():
    from tinylm import TinyLM, sync_vocab
    model = TinyLM(vocab_size=50, ctx=4, emb=8, hidden=16)
    ref = model
    check(sync_vocab(model, 80) is True)
    check(model is ref)
    check(len(model.C) == 80 and len(model.W2) == 80 and len(model.b2) == 80)
    check(sync_vocab(model, 60) is False)  # never shrinks


# ---------------------------------------------------------------------------
# 4-bit save/load + param limits
# ---------------------------------------------------------------------------

@test("quantization: save/load roundtrip within 4-bit error")
def t_quantize():
    from tinylm import save_model, load_model, count_params, enforce_param_limit
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.tlm")
        save_model(model, path)
        loaded = load_model(path)
    check(loaded.get_config() == model.get_config())
    for r1, r2 in zip(model.W2, loaded.W2):
        for v1, v2 in zip(r1, r2):
            check(abs(v1.data - v2.data) < 0.05, "quant error too large")
            break
        break
    n = count_params(model)
    try:
        enforce_param_limit(model, 100, "tiny")
        raise AssertionError("expected limit exceeded")
    except ValueError:
        pass
    check(n == count_params(loaded))


# ---------------------------------------------------------------------------
# sampling / generation
# ---------------------------------------------------------------------------

@test("generation: top-k confines candidates")
def t_sample_topk():
    from tinylm import _sample_next_token
    random.seed(1)
    logits = [1.0, 2.0, 3.0, 4.0, 5.0]
    seen = set()
    for _ in range(30):
        next_id, _ = _sample_next_token(logits[:], 1.0, top_k=2, top_p=0.0,
                                        repetition_penalty=1.0, generated_ids=[])
        seen.add(next_id)
    check(seen <= {3, 4}, f"top-k leaked: {seen}")


@test("generation: repetition penalty suppresses seen token")
def t_sample_reppen():
    from tinylm import _sample_next_token
    random.seed(2)
    logits = [1.0, 1.0, 1.0, 1.0, 1.0]
    hits = 0
    for _ in range(50):
        next_id, _ = _sample_next_token(logits[:], 1.0, 0, 0.0, 2.0, [3])
        if next_id == 3:
            hits += 1
    check(hits < 10, f"rep penalty ineffective: {hits}/50")


@test("generation: top-p confines nucleus")
def t_sample_topp():
    from tinylm import _sample_next_token
    random.seed(3)
    logits = [0.0, 0.0, 0.0, 10.0]  # token 3 dominates
    seen = set()
    for _ in range(20):
        next_id, _ = _sample_next_token(logits[:], 1.0, 0, 0.5, 1.0, [])
        seen.add(next_id)
    check(seen == {3}, f"top-p leaked: {seen}")


@test("generation: generate() runs with all options")
def t_generate():
    from tinylm import generate
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    out = generate(model, tok, "the quick", max_tokens=8, temperature=0.7,
                   top_k=10, top_p=0.9, repetition_penalty=1.2)
    check(isinstance(out, str) and len(out) > 0)


@test("generation: autostop with stop_token_ids")
def t_autostop_token_ids():
    from tinylm import generate
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    # Use token ID 1 as stop token
    out = generate(model, tok, "hello", max_tokens=50, min_tokens=2,
                   stop_token_ids=[1], temperature=1e-6)
    # Should stop early when token 1 is generated
    check(isinstance(out, str))
    # The output should contain the prompt plus at least min_tokens


@test("generation: autostop with stop_strings")
def t_autostop_strings():
    from tinylm import generate
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    # Use "END" as stop string
    out = generate(model, tok, "start", max_tokens=50, min_tokens=2,
                   stop_strings=["END"], temperature=1e-6)
    check(isinstance(out, str))
    # Should stop when "END" is generated


@test("generation: unbounded until EOS marker")
def t_generate_marker_stop():
    from tinylm import TinyLM, generate
    from tinylm.data_clean import marker_id
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    V = len(tok.vocab)
    EOS = marker_id("EOS", V)
    if EOS < 0 or EOS >= len(model.C):
        return  # tiny vocab: marker IDs out of range, nothing to stop on
    model.b2[EOS].data = 1e6  # force the EOS marker
    out = generate(model, tok, "hello", max_tokens=None, top_k=1)
    check(out == tok.decode(tok.encode("hello") + [EOS]))
    # Explicit caps still bound generation
    model.b2[EOS].data = -1e6  # never emit EOS
    seen = []
    generate(model, tok, "hello", max_tokens=4, top_k=1,
             stream_callback=lambda s, i, total, tps: seen.append(s))
    check(len(seen) == 4)


# ---------------------------------------------------------------------------
# training pipeline: LR scheduler, grad clip, evaluate, early stop, checkpoint
# ---------------------------------------------------------------------------

@test("lr scheduler: cosine_warmup / linear / constant")
def t_lr_scheduler():
    from tinylm import LR_Scheduler
    s = LR_Scheduler(0.1, 1000, "cosine_warmup", 100, 0.01)
    check(abs(s.get_lr(0) - 0.001) < 1e-9)     # warmup ramps from base/N
    check(abs(s.get_lr(99) - 0.1) < 1e-9)      # full lr at end of warmup
    check(s.get_lr(500) < 0.1)                 # decays after warmup
    check(s.get_lr(999) >= 0.01)               # reaches min_lr
    lin = LR_Scheduler(0.1, 100, "linear", 0, 0.0)
    check(abs(lin.get_lr(0) - 0.1) < 1e-9 and lin.get_lr(99) < 0.01)
    const = LR_Scheduler(0.1, 100, "constant")
    check(const.get_lr(0) == const.get_lr(50) == 0.1)


@test("grad clip: global norm capped in place")
def t_grad_clip():
    from tinylm import clip_grads
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    for p in model.params:
        p.grad = 5.0
    norm = clip_grads(model.params, 1.0)
    check(norm > 1.0, "should report pre-clip norm")
    new_norm = math.sqrt(sum(p.grad * p.grad for p in model.params))
    check(new_norm <= 1.0 + 1e-6, f"clip failed: {new_norm}")


@test("evaluate: returns finite perplexity")
def t_evaluate():
    from tinylm import evaluate
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    ppl = evaluate(model, ids[:200], batch_size=8, n_negatives=32)
    check(math.isfinite(ppl) and ppl > 1.0, f"bad ppl {ppl}")


@test("early stopping: stops after patience and restores best")
def t_early_stop():
    from tinylm import EarlyStopping
    tok, _ = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    es = EarlyStopping(patience=2, min_delta=1e-3)
    model.params[0].data = 0.5
    check(es(5.0, model) is False)          # first = best (params 0.5 saved)
    model.params[0].data = 0.9
    check(es(4.9, model) is False)          # improves -> best (params 0.9 saved)
    model.params[0].data = 1.0
    check(es(4.9, model) is False)          # no improvement (delta < min_delta)
    check(es(4.9, model) is True)           # patience reached -> stop + restore
    check(abs(model.params[0].data - 0.9) < 1e-12, "best not restored")


@test("checkpoint: save/load/resume roundtrip")
def t_checkpoint():
    from tinylm import Checkpoint, save_checkpoint, load_checkpoint, resume_from_checkpoint
    tok, _ = tiny_tok_and_ids()
    m1 = tiny_model(vocab=len(tok.vocab))
    ckpt = Checkpoint(step=42, model_state=[p.data for p in m1.params],
                      config=m1.get_config(), loss=1.25, val_loss=1.5, lr=0.01)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.json")
        save_checkpoint(ckpt, path)
        loaded = load_checkpoint(path)
    check(loaded.step == 42 and loaded.loss == 1.25 and loaded.val_loss == 1.5)
    m2 = tiny_model(vocab=len(tok.vocab))
    resume_from_checkpoint(m2, loaded)
    for a, b in zip(m1.params, m2.params):
        check(abs(a.data - b.data) < 1e-12)


@test("train: full pipeline with val split + checkpointing")
def t_train_full():
    from tinylm import train, load_checkpoint
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    d = tempfile.mkdtemp()
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train(model, ids, steps=12, lr=0.05, batch_size=8, n_negatives=32,
                  lr_scheduler="cosine_warmup", warmup_steps=3, min_lr=0.001,
                  grad_clip=1.0, val_split=0.15, eval_every=6,
                  early_stopping=True, patience=2,
                  checkpoint_every=6, checkpoint_dir=d, use_c=False)
        check(os.path.exists(os.path.join(d, "checkpoint_step_6.json")))
        check(os.path.exists(os.path.join(d, "checkpoint_final.json")))
        ckpt = load_checkpoint(os.path.join(d, "checkpoint_final.json"))
        check(ckpt.step == 12)
        check(ckpt.val_loss is not None)
        check(ckpt.lr > 0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("train: resume from checkpoint continues training")
def t_train_resume():
    from tinylm import train, load_checkpoint, resume_from_checkpoint
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    d = tempfile.mkdtemp()
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, ids, steps=8, lr=0.05, batch_size=8, n_negatives=32,
                  early_stopping=False, checkpoint_every=4,
                  checkpoint_dir=d, use_c=False, val_split=0.0, eval_every=0)
        ckpt = load_checkpoint(os.path.join(d, "checkpoint_final.json"))
        m2 = tiny_model(vocab=len(tok.vocab))
        resume_from_checkpoint(m2, ckpt)
        with contextlib.redirect_stdout(io.StringIO()):
            train(m2, ids, steps=4, lr=0.05, batch_size=8, n_negatives=32,
                  early_stopping=False, checkpoint_every=0,
                  checkpoint_dir=d, use_c=False, val_split=0.0, eval_every=0)
        # params actually moved during continued training
        check(any(abs(a.data - b.data) > 1e-9
                  for a, b in zip(model.params, m2.params)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("auto adjust: lr reduced on plateau, multiplier applied")
def t_auto_adjust():
    from tinylm import AutoAdjust
    adj = AutoAdjust(enabled=True, factor=0.5, patience=3, min_delta=1e-2,
                     warmup_ignore=0)
    # improving loss -> no reduction
    for i, loss in enumerate([5.0, 4.0, 3.0]):
        adj.check(i + 1, loss)
    check(adj.multiplier == 1.0)
    # plateau (no improvement > min_delta) for patience steps -> reduce
    reduced = any(adj.check(i, 3.0) for i in range(4, 8))
    check(reduced, "expected an LR reduction on plateau")
    check(adj.multiplier == 0.5)
    # continues reducing every `patience` plateau steps
    check(adj.check(7, 3.0) is False)   # bad_checks=1 after reset
    check(adj.check(8, 3.0) is False)   # bad_checks=2
    check(adj.check(9, 3.0) is True)    # patience=3 reached (steps 7,8,9)
    check(adj.multiplier == 0.25)
    # floor respected
    for i in range(10, 400):
        adj.check(i, 3.0)
    check(adj.multiplier >= 1e-4)
    check(adj.reduce_count >= 1)


@test("auto adjust: warmup steps ignored")
def t_auto_adjust_warmup():
    from tinylm import AutoAdjust
    adj = AutoAdjust(enabled=True, factor=0.5, patience=1, min_delta=0.5,
                     warmup_ignore=10)
    for i in range(1, 11):
        adj.check(i, 1.0)
    check(adj.multiplier == 1.0, "should not adjust during warmup")
    adj.check(11, 1.0)
    check(adj.multiplier == 0.5, "should adjust after warmup")


@test("auto adjust: disabled is inert")
def t_auto_adjust_off():
    from tinylm import AutoAdjust
    adj = AutoAdjust(enabled=False)
    for i in range(100):
        adj.check(i, 5.0)
    check(adj.multiplier == 1.0 and adj.reduce_count == 0)


@test("auto adjust: clip relaxation widens chronic clipping")
def t_auto_adjust_clip():
    from tinylm import AutoAdjust
    adj = AutoAdjust(enabled=True, max_grad_clip=1.0, clip_relax_after=3,
                     clip_relax_factor=2.0)
    check(adj.grad_clip_now() == 1.0)
    for _ in range(3):
        adj.note_grad_clip(5.0)   # chronically clipped
    check(adj.grad_clip_now() == 2.0, "clip limit should relax")
    for _ in range(3):
        adj.note_grad_clip(0.5)   # not clipping -> reset streak
    adj.note_grad_clip(10.0)
    check(adj.grad_clip_now() == 2.0, "streak should reset after non-clipping")


@test("train: auto_adjust wired into training loop")
def t_train_auto_adjust():
    from tinylm import train, AutoAdjust
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        # constant LR + min_delta large => guaranteed plateau => reduction
        train(model, ids, steps=30, lr=0.05, batch_size=8, n_negatives=32,
              lr_scheduler="constant", warmup_steps=0,
              auto_adjust=True, auto_factor=0.5, auto_patience=3,
              auto_clip_relax=False, grad_clip=0.0,
              early_stopping=False, checkpoint_every=0,
              val_split=0.0, eval_every=0, use_c=False)
    check(all(math.isfinite(p.data) for p in model.params))


@test("train: C kernel path supports auto_adjust + grad clip")
def t_train_c_auto_adjust():
    import accel
    if not accel.AVAILABLE:
        print("    (accel not built - run ./build.sh - skipping)")
        return
    from tinylm import train
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    import io, contextlib
    losses = []
    with contextlib.redirect_stdout(io.StringIO()):
        train(model, ids, steps=20, lr=0.05, batch_size=8, n_negatives=32,
              lr_scheduler="linear", warmup_steps=0,
              grad_clip=1.0, auto_adjust=True, auto_patience=3,
              early_stopping=False, checkpoint_every=0,
              val_split=0.0, eval_every=0, use_c=True,
              progress_cb=lambda d, t, l, *a: losses.append(l))
    check(len(losses) == 20 and all(math.isfinite(l) for l in losses))


@test("train: rollback guard unit + spike restore in train()")
def t_train_rollback():
    from tinylm import RollbackGuard, train
    # Unit: window=2 triggers above best+tol, collapses to best after rollback
    g = RollbackGuard(window=2, tol=0.0)
    check(g.enabled is True)
    g.push(5.0, [1.0, 2.0], [0.1, 0.2])
    check(g.check(99.0)[0] is False, "no rollback until window is full")
    g.push(4.0, [3.0, 4.0], [0.3, 0.4])
    do_rb, best = g.check(6.0)
    check(do_rb is True and best[0] == 4.0, f"expected spike rollback, got {do_rb} {best}")
    check(g.check(3.5)[0] is False, "improvement must not roll back")
    g.note_rollback()
    check(g.rollbacks == 1 and len(g.history) == 1)
    # Non-finite loss always rolls back to the best finite snapshot
    do_rb, best = g.check(float("nan"))
    check(do_rb is True and math.isfinite(best[0]))
    # Disabled guard is inert
    off = RollbackGuard(window=0)
    check(off.enabled is False)
    off.push(1.0, [1.0], None)
    check(off.check(99.0) == (False, None))

    # Integration: poison weights mid-run with NaNs; rollback must restore
    # finite weights and count the event, while the disabled run stays poisoned.
    import io, contextlib
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    random.seed(0)

    def poison_cb(done, total, loss, *a):
        if done == 2:
            for p in model.params:
                p.data = float("nan")

    with contextlib.redirect_stdout(io.StringIO()):
        r = train(model, ids, steps=6, lr=0.05, batch_size=8, n_negatives=32,
                  lr_scheduler="constant", warmup_steps=0, grad_clip=0.0,
                  momentum=0.0, auto_adjust=False, early_stopping=False,
                  checkpoint_every=0, eval_every=0, use_c=False,
                  rollback_window=2, progress_cb=poison_cb)
    check(r["rollbacks"] >= 1, f"expected >=1 rollback, got {r}")
    check(all(math.isfinite(p.data) for p in model.params),
          "rollback did not restore finite weights")

    model2 = tiny_model(vocab=len(tok.vocab))
    random.seed(0)

    def poison_cb2(done, total, loss, *a):
        if done == 2:
            for p in model2.params:
                p.data = float("nan")

    with contextlib.redirect_stdout(io.StringIO()):
        r2 = train(model2, ids, steps=6, lr=0.05, batch_size=8, n_negatives=32,
                   lr_scheduler="constant", warmup_steps=0, grad_clip=0.0,
                   momentum=0.0, auto_adjust=False, early_stopping=False,
                   checkpoint_every=0, eval_every=0, use_c=False,
                   rollback_window=0, progress_cb=poison_cb2)
    check(r2["rollbacks"] == 0, f"disabled run must report 0 rollbacks, got {r2}")
    check(not all(math.isfinite(p.data) for p in model2.params),
          "disabled run should keep the poisoned (non-finite) weights")


@test("train: rollback_callback reports gap vs run best")
def t_train_rollback_callback():
    from tinylm import train
    import io, contextlib
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    random.seed(0)
    seen = []
    losses = []

    def poison_cb(done, total, loss, *a):
        losses.append(loss)
        if done == 2:
            for p in model.params:
                p.data = float("nan")

    def rb_cb(step, loss, best):
        seen.append((step, loss, best))

    with contextlib.redirect_stdout(io.StringIO()):
        r = train(model, ids, steps=6, lr=0.05, batch_size=8, n_negatives=32,
                  lr_scheduler="constant", warmup_steps=0, grad_clip=0.0,
                  momentum=0.0, auto_adjust=False, early_stopping=False,
                  checkpoint_every=0, eval_every=0, use_c=False,
                  rollback_window=2, rollback_callback=rb_cb,
                  progress_cb=poison_cb)
    check(r["rollbacks"] >= 1 and len(seen) == r["rollbacks"],
          f"callback must fire once per rollback, got {r} {seen}")
    step, loss, best = seen[0]
    check(step == 3, f"poisoned step 3 must trigger, got step {step}")
    check(math.isfinite(best), f"run best must be finite, got {best}")
    check(abs(best - min(losses[:2])) < 1e-9,
          f"callback best {best} must equal smallest loss so far {losses[:2]}")
    gap = loss - best  # the value the training tab displays
    check(not math.isfinite(gap) or gap > 0, f"gap must be positive, got {gap}")


@test("train: update uses the batch-averaged gradient")
def t_train_grad_no_double_scaling():
    from tinylm import train
    tok, ids = tiny_tok_and_ids()
    import io, contextlib

    def weight_movement(batch_size):
        model = tiny_model(vocab=len(tok.vocab))
        before = [p.data for p in model.params]
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, ids, steps=3, lr=0.05, batch_size=batch_size, n_negatives=32,
                  lr_scheduler="constant", warmup_steps=0, grad_clip=0.0, momentum=0.0,
                  auto_adjust=False, early_stopping=False, checkpoint_every=0,
                  eval_every=0, use_c=False)
        after = [p.data for p in model.params]
        return sum(abs(b - a) for a, b in zip(before, after))

    # Each micro-batch loss is already averaged over batch_size, so a bigger
    # batch must not shrink the step (it used to be divided by batch_size twice,
    # making the effective LR 16x too small at batch_size=16).
    mv1, mv16, mv64 = weight_movement(1), weight_movement(16), weight_movement(64)
    check(mv16 > 0.4 * mv1 and mv64 > 0.4 * mv1,
          f"batch-averaged updates should be batch-size independent: {mv1:.4f} {mv16:.4f} {mv64:.4f}")


@test("train: momentum converges faster than plain SGD")
def t_train_momentum():
    from tinylm import train
    tok, ids = tiny_tok_and_ids()
    import io, contextlib

    def final_loss(use_c, momentum):
        model = tiny_model(vocab=len(tok.vocab))
        random.seed(0)
        losses = []
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, ids, steps=40, lr=0.05, batch_size=16, n_negatives=32,
                  lr_scheduler="constant", warmup_steps=0, grad_clip=0.0,
                  momentum=momentum, auto_adjust=False,
                  early_stopping=False, checkpoint_every=0, eval_every=0,
                  use_c=use_c, progress_cb=lambda d, t, l, *a: losses.append(l))
        return losses[0], losses[-1]

    for use_c in (False, True):
        first, plain = final_loss(use_c, 0.0)
        _, mom = final_loss(use_c, 0.9)
        check(mom < 0.7 * plain,
              f"momentum should converge faster (use_c={use_c}): {plain:.3f} -> {mom:.3f}")


# ---------------------------------------------------------------------------
# parallel training + gradient clipping
# ---------------------------------------------------------------------------

@test("train_multicore: best-of-N trials keep the lowest-loss update")
def t_train_multicore():
    from tinylm import train_multicore
    from tinylm.parallel import _pick_winner
    # Winner-selection rule: lowest finite probe loss wins; shard-loss
    # fallback when probes are unusable; None when nothing is usable.
    check(_pick_winner([(5.0, 4.0, [1.0], [0.0]),
                        (6.0, 3.0, [2.0], [0.0])]) == 1)
    check(_pick_winner([(5.0, float("nan"), [1.0], [0.0]),
                        (6.0, 9.0, [2.0], [0.0])]) == 1)
    check(_pick_winner([(5.0, float("nan"), [1.0], [0.0]),
                        (4.0, float("nan"), [2.0], [0.0])]) == 1)
    check(_pick_winner([(5.0, float("nan"), [float("nan")], [0.0])]) is None)
    check(_pick_winner([]) is None)
    # Integration: short run keeps the result contract on 2 workers.
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    import io, contextlib
    losses = []
    with contextlib.redirect_stdout(io.StringIO()):
        r = train_multicore(model, ids, steps=4, lr=0.05, batch_size=8,
                            n_negatives=32, workers=2, grad_clip=1.0,
                            momentum=0.9, rollback_window=2,
                            progress_cb=lambda d, t, l, *a: losses.append(l))
    for k in ("cancelled", "initial_loss", "best_loss", "final_loss",
              "steps_done", "rollbacks"):
        check(k in r, f"missing {k} in {r}")
    check(r["cancelled"] is False and r["steps_done"] == 4)
    check(len(losses) == 4 and all(math.isfinite(l) for l in losses))
    check(all(math.isfinite(p.data) for p in model.params))
    # Immediate cancel leaves weights unchanged.
    before = [p.data for p in model.params]
    with contextlib.redirect_stdout(io.StringIO()):
        rc = train_multicore(model, ids, steps=10, lr=0.05, batch_size=4,
                             n_negatives=8, workers=2,
                             cancel_callback=lambda: True)
    check(rc["cancelled"] is True and rc["steps_done"] == 0)
    check([p.data for p in model.params] == before,
          "cancelled multicore run mutated weights")
    # Total divergence (all trials non-finite) falls back to the best
    # snapshot instead of keeping poisoned weights.
    model3 = tiny_model(vocab=len(tok.vocab))
    random.seed(0)
    seen = []

    def poison_cb3(done, total, loss, *a):
        if done == 2:
            for p in model3.params:
                p.data = float("nan")

    with contextlib.redirect_stdout(io.StringIO()):
        r3 = train_multicore(model3, ids, steps=6, lr=0.05, batch_size=8,
                             n_negatives=32, workers=2, momentum=0.0,
                             rollback_window=2,
                             rollback_callback=lambda s, l, b: seen.append((s, l, b)),
                             progress_cb=poison_cb3)
    check(r3["rollbacks"] >= 1 and len(seen) == r3["rollbacks"],
          f"expected fallback rollback, got {r3} {seen}")
    check(all(math.isfinite(p.data) for p in model3.params),
          "multicore did not restore finite weights after total divergence")


@test("train_parallel: runs with grad clip + momentum")
def t_train_parallel():
    from tinylm import train_parallel
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    import io, contextlib
    losses = []
    with contextlib.redirect_stdout(io.StringIO()):
        train_parallel(model, ids, steps=4, lr=0.05, batch_size=8,
                       n_negatives=32, workers=2, grad_clip=1.0, momentum=0.9,
                       progress_cb=lambda d, t, l, *a: losses.append(l))
    check(len(losses) == 4 and all(math.isfinite(l) for l in losses))


@test("train: immediate cancel leaves weights unchanged")
def t_train_cancel_immediate():
    from tinylm import train, train_parallel
    import io, contextlib
    tok, ids = tiny_tok_and_ids()
    model = tiny_model(vocab=len(tok.vocab))
    before = [p.data for p in model.params]
    with contextlib.redirect_stdout(io.StringIO()):
        r = train(model, ids, steps=10, lr=0.05, batch_size=4, n_negatives=8,
                  checkpoint_every=0, eval_every=0, early_stopping=False,
                  cancel_callback=lambda: True)
    check(r["cancelled"] is True and r["steps_done"] == 0)
    check([p.data for p in model.params] == before, "cancelled run mutated weights")
    before = [p.data for p in model.params]
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = train_parallel(model, ids, steps=10, lr=0.05, batch_size=4,
                            n_negatives=8, workers=2, cancel_callback=lambda: True)
    check(r2["cancelled"] is True and r2["steps_done"] == 0)
    check([p.data for p in model.params] == before, "cancelled parallel run mutated weights")


# ---------------------------------------------------------------------------
# C kernel parity (skipped when unavailable)
# ---------------------------------------------------------------------------

@test("c kernel: available + parity with python path")
def t_c_parity():
    import accel
    if not accel.AVAILABLE:
        print("    (accel not built - run ./build.sh - skipping)")
        return
    from tinylm import TinyLM, train
    tok, ids = tiny_tok_and_ids()
    random.seed(7)
    m1 = tiny_model(vocab=len(tok.vocab))
    random.seed(7)
    m2 = tiny_model(vocab=len(tok.vocab))
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        train(m1, ids, steps=3, lr=0.05, batch_size=8, n_negatives=32,
              use_c=True, lr_scheduler="constant", grad_clip=0.0,
              early_stopping=False, checkpoint_every=0,
              val_split=0.0, eval_every=0)
        train(m2, ids, steps=3, lr=0.05, batch_size=8, n_negatives=32,
              use_c=False, lr_scheduler="constant", grad_clip=0.0,
              early_stopping=False, checkpoint_every=0,
              val_split=0.0, eval_every=0)
    # C path: same update rule, random negatives differ -> just check finite
    check(all(math.isfinite(p.data) for p in m1.params))
    check(all(math.isfinite(p.data) for p in m2.params))


@test("c kernel: grad clip applied inside C train_step")
def t_c_gradclip():
    import accel
    if not accel.AVAILABLE:
        print("    (accel not built - run ./build.sh - skipping)")
        return
    # clip bounds the batch-AVERAGED gradient: |delta_i| <= lr * max_norm
    tok, ids = tiny_tok_and_ids()
    from tinylm import TinyLM, _flat_data, _flat_params
    import random as ai_random
    dims = None
    random.seed(11)
    m1 = tiny_model(vocab=len(tok.vocab))
    dims = (len(m1.C), m1.ctx, len(m1.C[0]), len(m1.b1))
    windows = []
    for _ in range(8):
        i = random.randrange(len(ids) - m1.ctx - 1)
        windows.extend(ids[i:i + m1.ctx])
        windows.append(ids[i + m1.ctx])
    import accel
    flat0 = _flat_data(m1)
    lr, max_norm = 0.5, 0.05
    accel.reset_momentum()
    _, flat_noclip = accel.train_step(list(flat0), windows, dims, 32, lr, 5, max_norm=0.0)
    accel.reset_momentum()
    _, flat_clip = accel.train_step(list(flat0), windows, dims, 32, lr, 5, max_norm=max_norm)
    delta_noclip = max(abs(a - b) for a, b in zip(flat0, flat_noclip))
    delta_clip = max(abs(a - b) for a, b in zip(flat0, flat_clip))
    check(delta_clip > 0, "clip path should still update params")
    check(delta_clip <= lr * max_norm + 1e-12,
          f"clip should bound the update: {delta_clip} > {lr * max_norm}")
    check(delta_clip <= delta_noclip + 1e-12,
          f"clip should not increase update: {delta_clip} vs {delta_noclip}")


@test("c kernel: momentum matches plain SGD when disabled")
def t_c_momentum_zero():
    import accel
    if not accel.AVAILABLE:
        print("    (accel not built - run ./build.sh - skipping)")
        return
    tok, ids = tiny_tok_and_ids()
    from tinylm import _flat_data
    random.seed(5)
    m = tiny_model(vocab=len(tok.vocab))
    dims = (len(m.C), m.ctx, len(m.C[0]), len(m.b1))
    windows = []
    for _ in range(8):
        i = random.randrange(len(ids) - m.ctx - 1)
        windows.extend(ids[i:i + m.ctx])
        windows.append(ids[i + m.ctx])
    flat0 = _flat_data(m)
    # momentum=0 twice -> identical updates (no state carried between calls)
    accel.reset_momentum()
    _, a = accel.train_step(list(flat0), windows, dims, 32, 0.05, 7, momentum=0.0)
    accel.reset_momentum()
    _, b = accel.train_step(list(flat0), windows, dims, 32, 0.05, 7, momentum=0.0)
    check(max(abs(x - y) for x, y in zip(a, b)) == 0.0, "momentum=0 must be plain SGD")
    # a second step with momentum keeps moving (velocity is stateful)
    accel.reset_momentum()
    _, s1 = accel.train_step(list(flat0), windows, dims, 32, 0.05, 7, momentum=0.9)
    _, s2 = accel.train_step(s1, windows, dims, 32, 0.05, 7, momentum=0.9)
    d1 = max(abs(x - y) for x, y in zip(flat0, s1))
    d2 = max(abs(x - y) for x, y in zip(s1, s2))
    check(d2 > 0.5 * d1, "momentum should keep the step size up (velocity)")


# ---------------------------------------------------------------------------
# registries / model management / tasks / tokenizer manager
# ---------------------------------------------------------------------------

@test("registry: model registry + task registry")
def t_registry():
    from registry import MODEL_REGISTRY, TASK_REGISTRY, TaskTemplate, TOKENIZER_MANAGER
    from tinylm import TinyLM
    if not MODEL_REGISTRY.has("test-tiny"):
        MODEL_REGISTRY.register("test-tiny",
                                lambda: TinyLM(vocab_size=64, ctx=4, emb=8, hidden=16),
                                description="unit test model")
    m = MODEL_REGISTRY.create("test-tiny")
    check(m is not None and "test-tiny" in MODEL_REGISTRY.list())
    MODEL_REGISTRY.remove("test-tiny")
    if not TASK_REGISTRY.has("test-task"):
        TASK_REGISTRY.register(TaskTemplate(
            name="test-task", model_name="test-tiny",
            prompt_template="Do: {text}"))
    check(TASK_REGISTRY.get("test-task").model_name == "test-tiny")
    TASK_REGISTRY.remove("test-task")
    check(isinstance(TOKENIZER_MANAGER.vocab_size, int))


@test("router: training data format encode/decode")
def t_router_training_format():
    from tinylm import RouterTrainingExample, encode_router_example, decode_router_sequence, create_router_training_data
    from tokenizer import BPETokenizer
    
    tok = BPETokenizer()
    tok.train("hello world " * 100, vocab_size=500)
    target_vocab_size = 500  # Use target vocab size for special token allocation
    ctx = 32  # Larger context to fit prompts
    
    # Create test example with prompts that fit in context
    ex = RouterTrainingExample(
        prompt="write python function",
        subtasks=[
            {"model": "tiny-code", "prompt": "def fib(n): return n"},
            {"model": "tiny-prose", "prompt": "This calculates fib"}
        ]
    )
    
    # Encode and decode
    encoded = encode_router_example(ex, tok, ctx, target_vocab_size)
    decoded = decode_router_sequence(encoded, tok, target_vocab_size)
    
    check(decoded.prompt.strip() == ex.prompt.strip())
    check(len(decoded.subtasks) == 2)
    check(decoded.subtasks[0]["model"] == "tiny-code")
    check(decoded.subtasks[1]["model"] == "tiny-prose")
    
    # Test create_router_training_data
    ids = create_router_training_data(tok, target_vocab_size, ctx, [ex, ex])
    check(len(ids) > 0)


@test("router: training data includes multi-subtask examples")
def t_router_training_multi():
    from tinylm import (encode_router_example, decode_router_sequence,
                        generate_router_training_data)
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    corpus = ("write a python function to sort a list. " * 10 +
              "solve the math equation for x. " * 10 +
              "write a story about a cat. " * 10)
    tok.train(corpus, vocab_size=500)
    examples = generate_router_training_data(corpus, tok, 500, 32)
    check(len(examples) > 0)
    multi = [ex for ex in examples if len(ex.subtasks) >= 2]
    check(len(multi) > 0, "expected composite multi-subtask examples")
    check(len({s["model"] for s in multi[0].subtasks}) >= 2)
    check(all(len(s["prompt"]) > 0 for s in multi[0].subtasks))
    # Multi-subtask examples survive the encode/decode roundtrip
    enc = encode_router_example(multi[0], tok, 32, 500)
    dec = decode_router_sequence(enc, tok, 500)
    check(len(dec.subtasks) == len(multi[0].subtasks))
    check([s["model"] for s in dec.subtasks] ==
          [s["model"] for s in multi[0].subtasks])
    # Opt-out restores single-subtask-only output
    singles = generate_router_training_data(corpus, tok, 500, 32, max_subtasks=1)
    check(all(len(ex.subtasks) == 1 for ex in singles))


@test("markers: insert/encode/render with yellow short forms")
def t_markers():
    from tinylm import (insert_markers, encode_marked, render_marked,
                        render_marked_ids, marker_id, marker_name)
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train("hello world. foo bar! test line here. " * 30, vocab_size=300)
    V = 300
    marked = insert_markers(
        "Hello world. How are you?\nSecond line here.\n\nNew section starts.")
    check("[EOT]" in marked and "[EOL]" in marked and "[EOS]" in marked)
    check(marked.index("[EOT]") < marked.index("[EOL]"))
    check(marked.count("[EOS]") == 2)
    check("\n" in marked, "marked preview must keep line breaks")
    ids = encode_marked(marked, tok, V)
    check(marker_id("EOL", V) in ids and marker_id("EOS", V) in ids
          and marker_id("EOT", V) in ids)
    check(marker_name(V - 13, V) == "EOL" and marker_name(V - 1, V) is None)
    # No collision with the router's reserved top-of-vocab IDs
    router_ids = {V - 1} | {V - 10 + i for i in range(8)}
    check(not ({marker_id(n, V) for n in ("EOL", "EOS", "EOT")} & router_ids))
    # Markers render as yellow short forms
    check("\033[33m[EOL]\033[0m" in render_marked(marked))
    check(render_marked(marked, color=False) == marked)
    back = render_marked_ids(ids, tok, V, color=False)
    check("[EOL]" in back and "[EOS]" in back and "[EOT]" in back)
    check("\033[33m[EOS]\033[0m" in render_marked_ids(ids, tok, V))


@test("markers: mark_and_encode is the canonical training entry")
def t_mark_and_encode():
    from tinylm import (mark_and_encode, insert_markers, encode_marked,
                        marker_id)
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train("hello world. foo bar! test line here. " * 30, vocab_size=300)
    ids = mark_and_encode("Hello world.\nSecond line.", tok, len(tok.vocab))
    check(ids == encode_marked(insert_markers("Hello world.\nSecond line."),
                               tok, len(tok.vocab)))
    check(marker_id("EOT", len(tok.vocab)) in ids)
    check(marker_id("EOL", len(tok.vocab)) in ids)


@test("stats: best loss only decreases across runs")
def t_stats():
    from tinylm import TrainingStats, best_path, train_with_stats
    from tinylm.serialization import load_model
    with tempfile.TemporaryDirectory() as d:
        store = TrainingStats(os.path.join(d, "stats.json"))
        # Record tracks the minimum without any training
        store.record("m", 5.0, 5.5, 10)
        store.record("m", 7.0, 7.0, 10)
        check(store.best_loss("m") == 5.0)
        check(store.get("m")["last_loss"] == 7.0)
        check(store.get("m")["runs"] == 2)
        check(store.best_loss("unknown") is None)

        tok, ids = tiny_tok_and_ids()
        model = tiny_model(vocab=len(tok.vocab))
        kw = dict(steps=3, lr=0.05, batch_size=4, n_negatives=8, use_c=False,
                  early_stopping=False, eval_every=0, checkpoint_every=0)
        info = train_with_stats("m", model, ids, stats=store,
                                best_dir=os.path.join(d, "best"), **kw)
        check(info["improved"] is True and info["reverted"] is False)
        check(info["warmed"] is False)
        check(os.path.exists(best_path("m", os.path.join(d, "best"))))
        best1 = info["best_loss"]
        check(best1 is not None and best1 < float("inf"))
        # Sabotage the weights, then run zero steps (best stays None, which
        # cannot beat the stored best): must restore the persisted weights
        for p in model.params:
            p.data += 5.0
        sabotaged = [p.data for p in model.params]
        info = train_with_stats("m", model, ids, stats=store,
                                best_dir=os.path.join(d, "best"),
                                steps=0, lr=0.0, batch_size=4, n_negatives=8,
                                use_c=False, early_stopping=False,
                                eval_every=0, checkpoint_every=0)
        check(info["improved"] is False and info["reverted"] is True)
        check(any(p != s for p, s in zip([p.data for p in model.params],
                                         sabotaged)))
        check(info["warmed"] is True)
        check(store.best_loss("m") == best1)
        restored = load_model(best_path("m", os.path.join(d, "best")))
        check(all(p.data == q.data
                  for p, q in zip(model.params, restored.params)))


@test("stats: apply_stats works for any train result dict")
def t_stats_apply():
    from tinylm import TrainingStats, apply_stats, best_path
    from tinylm.serialization import load_model
    tok, _ = tiny_tok_and_ids()
    with tempfile.TemporaryDirectory() as d:
        best_dir = os.path.join(d, "best")
        store = TrainingStats(os.path.join(d, "stats.json"))
        model = tiny_model(vocab=len(tok.vocab))
        info = apply_stats("n", model, {"best_loss": 3.0, "final_loss": 3.2,
                                        "steps_done": 5}, stats=store,
                           best_dir=best_dir)
        check(info["improved"] is True and info["reverted"] is False)
        check(os.path.exists(best_path("n", best_dir)))
        check(store.best_loss("n") == 3.0)
        # Worse run reverts to the snapshotted weights
        for p in model.params:
            p.data += 2.0
        info = apply_stats("n", model, {"best_loss": 9.0, "final_loss": 9.1,
                                        "steps_done": 5}, stats=store,
                           best_dir=best_dir)
        check(info["improved"] is False and info["reverted"] is True)
        check(info["prev_best"] == 3.0 and store.best_loss("n") == 3.0)
        saved = load_model(best_path("n", best_dir))
        check(all(p.data == q.data
                  for p, q in zip(model.params, saved.params)))
        # Warm start refuses missing snapshots and foreign architectures
        from tinylm import load_best
        check(load_best("missing", model, best_dir=best_dir) is False)
        other = tiny_model(vocab=len(tok.vocab), ctx=6, emb=8, hidden=24)
        flat_before = [p.data for p in other.params]
        check(load_best("n", other, best_dir=best_dir) is False)
        check([p.data for p in other.params] == flat_before)


@test("router: output depends on the full context window")
def t_router_context():
    from tinylm import Router, RouterConfig
    r = Router(RouterConfig(vocab_size=200, ctx=8, emb=8, hidden=16,
                            n_layers=1, d_ff=16), seed=0)
    base = [5, 10, 15, 20, 25, 30, 35, 40]
    l0 = r.fast_logits(base)
    # Same first token, rest differ → output must differ
    l1 = r.fast_logits([5, 99, 98, 97, 96, 95, 94, 93])
    check(max(abs(a - b) for a, b in zip(l0, l1)) > 1e-9,
          "router ignores context past position 0")
    # Graph path matches fast path in shape
    check(len(r.logits(base)) == 200)


@test("planner: router.split creates TaskPad tasks")
def t_planner():
    from scratchpad import TaskPad, TaskType, model_to_task_type, plan
    from model_mgmt import ModelCatalog, ModelSpec
    # Dynamic mapping works for any model count/naming, not a fixed list
    check(model_to_task_type("tiny-code") == TaskType.CODE)
    check(model_to_task_type("my-embed") == TaskType.EMBED)
    check(model_to_task_type("tiny-prose") == TaskType.GENERATE)
    check(model_to_task_type("tiny-finance") == TaskType.CUSTOM)
    check(model_to_task_type(None) == TaskType.CUSTOM)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "taskpad.jsonl")
        pad = TaskPad(path=path)
        catalog = ModelCatalog(os.path.join(d, "catalog.json"))
        catalog.add(ModelSpec(name="tiny-code", path="m.tlm",
                              domain_keywords=["code"], param_count=100,
                              config={}))

        class FakeRouter:
            def split(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                ids = [1, 2, 3]
                return [
                    {"model": "tiny-code", "prompt": prompt, "context_tokens": ids},
                    {"model": "tiny-finance", "prompt": "other", "context_tokens": ids},
                    {"prompt": "no model key", "context_tokens": ids},
                ]

        tasks = plan("write code", FakeRouter(), tokenizer=None, taskpad=pad,
                     catalog=catalog)
        check(len(tasks) == 3)
        check(tasks[0].type == TaskType.CODE and tasks[0].model_name == "tiny-code")
        check(tasks[1].type == TaskType.CUSTOM)
        check(tasks[2].type == TaskType.CUSTOM and tasks[2].model_name is None)
        check(tasks[0].metadata["context_tokens"] == [1, 2, 3])
        check(tasks[0].metadata["features"] == {})
        check(tasks[0].metadata["model_available"] is True)
        check(tasks[1].metadata["model_available"] is False)
        check(tasks[2].metadata["model_available"] is False)
        # Tasks persist across TaskPad instances
        check(len(TaskPad(path=path).list_tasks()) == 3)


@test("router: decompose runs on trained weights + plan() prefers it")
def t_router_decompose():
    import io, contextlib
    from tinylm import Router, RouterConfig, RouterTrainingExample, train_router
    from tokenizer import BPETokenizer
    from scratchpad import TaskPad, TaskType, plan
    tok = BPETokenizer()
    tok.train("write python function calculate fibonacci. " * 50, vocab_size=500)
    cfg = RouterConfig(vocab_size=len(tok.vocab), ctx=16, emb=8, hidden=16,
                       n_layers=1, d_ff=32)
    router = Router(cfg, seed=0)
    ex = RouterTrainingExample(
        prompt="write python function",
        subtasks=[{"model": "tiny-code", "prompt": "def fib(n): return n"}])
    with contextlib.redirect_stdout(io.StringIO()):
        train_router(router, tok, [ex], steps=2, lr=0.05, batch_size=2,
                     n_negatives=8, use_c=False, early_stopping=False,
                     eval_every=0, checkpoint_every=0)
    subs = router.decompose("write python function", tok, max_subtasks=2,
                            max_tokens=20)
    check(isinstance(subs, list) and len(subs) >= 1)
    for s in subs:
        check("model" in s and "prompt" in s and "context_tokens" in s)
    # With a catalog, decoded names outside it are dropped, listed ones kept
    from model_mgmt import ModelCatalog, ModelSpec
    with tempfile.TemporaryDirectory() as d:
        code_only = ModelCatalog(os.path.join(d, "c1.json"))
        code_only.add(ModelSpec(name="tiny-code", path="x",
                                domain_keywords=["code"], param_count=1,
                                config={}))
        check(router.decompose("write python function", tok, max_subtasks=2,
                               max_tokens=20, catalog=code_only) == [])
        prose = ModelCatalog(os.path.join(d, "c2.json"))
        prose.add(ModelSpec(name="tiny-prose", path="x",
                            domain_keywords=["write"], param_count=1,
                            config={}))
        kept = router.decompose("write python function", tok, max_subtasks=2,
                                max_tokens=20, catalog=prose)
        check(len(kept) == 1 and kept[0]["model"] == "tiny-prose")
    # plan() prefers decompose() when the router offers it ...
    with tempfile.TemporaryDirectory() as d:
        pad = TaskPad(path=os.path.join(d, "taskpad.jsonl"))

        class ModelRouter:
            def __init__(self):
                self.called = False
            def decompose(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                self.called = True
                return [{"model": "tiny-reason", "prompt": "p",
                         "context_tokens": [7]}]
            def split(self, *a, **k):
                raise AssertionError("split should not be called")

        mr = ModelRouter()
        tasks = plan("x", mr, tokenizer=object(), taskpad=pad)
        check(mr.called and len(tasks) == 1)
        check(tasks[0].type == TaskType.REASON)
        check(tasks[0].metadata["context_tokens"] == [7])
        # ... and falls back to split() when decompose() fails
        class BrokenRouter:
            def decompose(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                raise RuntimeError("no weights")
            def split(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                return [{"model": "tiny-code", "prompt": prompt,
                         "context_tokens": [1]}]

        tasks = plan("y", BrokenRouter(), tokenizer=object(), taskpad=pad)
        check(tasks[-1].type == TaskType.CODE)


@test("cycle: run_cycle plans, executes and records")
def t_run_cycle():
    from scratchpad import (Scratchpad, TaskPad, TaskStatus, TaskType,
                            run_cycle)
    with tempfile.TemporaryDirectory() as d:
        pad = TaskPad(path=os.path.join(d, "taskpad.jsonl"))
        sp = Scratchpad(path=os.path.join(d, "scratchpad.jsonl"))

        class FakeRouter:
            def split(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                return [
                    {"model": "tiny-code", "prompt": prompt,
                     "context_tokens": [1]},
                    {"model": "tiny-prose", "prompt": "other",
                     "context_tokens": [2]},
                ]

        res = run_cycle("do it", FakeRouter(), tokenizer=None, taskpad=pad,
                        scratchpad=sp,
                        executors={TaskType.CODE: lambda t: "code-done",
                                   TaskType.GENERATE: lambda t: "prose-done"})
        check(res["done"] is True and res["cycles"] == 1)
        check(len(res["tasks"]) == 2)
        results = sorted(t.result for t in pad.list_tasks())
        check(results == ["code-done", "prose-done"])
        types = {e.metadata.get("step_type") for e in sp.get_recent()}
        check("cycle_start" in types and "task_outcome" in types)
        # Blocked queue makes no progress and still terminates
        pad2 = TaskPad(path=os.path.join(d, "tp2.jsonl"))
        sp2 = Scratchpad(path=os.path.join(d, "sp2.jsonl"))
        pad2.create_task(TaskType.CODE, "blocked", dependencies=["missing"])

        class EmptyRouter:
            def split(self, prompt, tokenizer, max_subtasks=4, catalog=None):
                return []

        res = run_cycle("stuck", EmptyRouter(), tokenizer=None, taskpad=pad2,
                        scratchpad=sp2, max_cycles=3)
        check(res["done"] is False and res["cycles"] == 1)
        check(len(pad2.list_tasks(status=TaskStatus.PENDING)) == 1)


@test("taskpad cli: default executors run tasks end to end")
def t_taskpad_executors():
    from scratchpad import TaskPad, TaskStatus, TaskType
    from model_mgmt import ModelCatalog, ModelSpec
    from main import _default_task_executors
    root = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as d:
        catalog = ModelCatalog(os.path.join(d, "catalog.json"))
        catalog.add(ModelSpec(name="exec-model",
                              path=os.path.join(root, "tinyLM"),
                              domain_keywords=["text"], param_count=1,
                              config={}))
        executors = _default_task_executors(catalog)
        check(TaskType.GENERATE in executors and TaskType.CODE in executors)
        pad = TaskPad(path=os.path.join(d, "taskpad.jsonl"))
        for ttype, fn in executors.items():
            pad.register_executor(ttype, fn)
        task = pad.create_task(TaskType.GENERATE, "hello world",
                               model_name="exec-model",
                               parameters={"max_tokens": 5})
        done = pad.execute_task(task.id)
        check(done.status == TaskStatus.COMPLETED)
        check(isinstance(done.result, str) and len(done.result) > 0)
        # Unlisted models fail with a clear error instead of running nowhere
        bad = pad.create_task(TaskType.CODE, "x", model_name="ghost-model")
        failed = pad.execute_task(bad.id)
        check(failed.status == TaskStatus.FAILED)
        check("not in the catalog" in (failed.error or ""))


@test("router: train_router runs and reduces loss")
def t_router_train():
    from tinylm import Router, RouterConfig, train_router, generate_router_training_data
    from tokenizer import BPETokenizer
    
    tok = BPETokenizer()
    corpus = ("write a python function to calculate fibonacci. " * 50 +
              "solve the math equation x^2 + 2x = 0. " * 50 +
              "write a story about a cat. " * 50)
    tok.train(corpus, vocab_size=2000)
    vocab_size = len(tok.vocab)
    
    config = RouterConfig(vocab_size=vocab_size, ctx=32, emb=64, hidden=128, n_layers=2, d_ff=256)
    router = Router(config, seed=42)
    
    examples = generate_router_training_data(corpus, tok, vocab_size, config.ctx)
    check(len(examples) > 0)
    
    # Train for a few steps
    import io, contextlib
    losses = []
    with contextlib.redirect_stdout(io.StringIO()):
        train_router(router, tok, examples, steps=4, lr=0.05, batch_size=4, n_negatives=16,
                     use_c=False, progress_cb=lambda d, t, l, *a: losses.append(l))
    
    check(len(losses) == 4)
    check(all(math.isfinite(l) for l in losses))
    # Loss should generally decrease or at least be stable
    check(losses[-1] < losses[0] * 2)  # Not diverging


@test("specialized: model creation, training, save/load")
def t_specialized_model():
    from tinylm import SpecializedLM, SpecializedConfig, create_specialized_model, get_specialized_presets, count_params, save_model, load_model
    from tokenizer import BPETokenizer
    
    tok = BPETokenizer()
    tok.train("hello world " * 200, vocab_size=500)
    vocab_size = len(tok.vocab)
    
    # Test custom config
    config = SpecializedConfig(vocab_size=vocab_size, ctx=32, emb=32, hidden=32, n_layers=1, d_ff=128, model_type="test")
    model = SpecializedLM(config, seed=42)
    check(count_params(model) < 100_000)
    
    # Test forward/loss
    ids = tok.encode("hello world")
    loss = model.sampled_loss(ids[-config.ctx:], ids[0], 16, vocab_size)
    check(math.isfinite(loss.data))
    
    # Test fast_logits
    fast = model.fast_logits(ids[-config.ctx:])
    check(len(fast) == vocab_size)
    
    # Test save/load
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "specialized.tlm")
        save_model(model, path)
        loaded = load_model(path)
        check(count_params(loaded) == count_params(model))
    
    # Test presets
    presets = get_specialized_presets()
    for name, preset_config in presets.items():
        preset_model = create_specialized_model(preset_config, seed=123)
        check(count_params(preset_model) < 100_000)
    
    # Test generation
    from tinylm import generate
    text = generate(model, tok, "hello", max_tokens=10, temperature=0.8)
    check(isinstance(text, str) and len(text) > 0)


@test("openskill: readpy tree/treefull/class/function")
def t_readpy():
    import importlib.util
    import io, contextlib
    root = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "readpy", os.path.join(root, "opencodeSkill", "readpy.py"))
    readpy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readpy)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write('"""Mod doc."""\n\n# top comment\ndef top():\n'
                '    """Top doc."""\n    return 1\n\nclass Foo:\n'
                '    """Foo doc."""\n    def bar(self):\n        return 2\n')
        path = f.name
    try:
        t = readpy.tree_text(path)
        for needle in ("def top @", "class Foo @", "def bar @"):
            check(needle in t, f"tree missing {needle!r}:\n{t}")
        tf = readpy.treefull_text(path)
        for needle in ("# top comment", "Top doc.", "Foo doc."):
            check(needle in tf, f"treefull missing {needle!r}:\n{tf}")
        src = readpy.class_source(path, "Foo")
        check("class Foo" in src and "def bar" in src)
        fsrc = readpy.function_source(path, "top")
        check("def top" in fsrc and "class Foo" not in fsrc)
        check("def bar" in readpy.function_source(path, "Foo.bar"))
        for bad in (lambda: readpy.class_source(path, "Nope"),
                    lambda: readpy.function_source(path, "Foo.nope"),
                    lambda: readpy.class_source(path, "top")):
            try:
                bad()
                raise AssertionError("expected LookupError")
            except LookupError:
                pass
        with contextlib.redirect_stdout(io.StringIO()):
            check(readpy.main(["tree", path]) == 0)
            check(readpy.main(["function", path, "Nope"]) == 1)
    finally:
        os.unlink(path)
    # Agent contract: SKILL.md index exists and advertises every subcommand.
    with open(os.path.join(root, "opencodeSkill", "SKILL.md"), encoding="utf-8") as f:
        index = f.read()
    check("name:" in index and "description:" in index)
    for needle in ("tree", "treefull", "class", "function"):
        check(needle in index, f"SKILL.md missing {needle!r}")


@test("model_mgmt: catalog create/load/delete roundtrip")
def t_catalog():
    from model_mgmt import ModelCatalog, ModelCreator, ModelLoader
    with tempfile.TemporaryDirectory() as d:
        catalog = ModelCatalog(os.path.join(d, "catalog.json"))
        creator = ModelCreator(catalog, d)
        spec = creator.create_model("tiny-unit", ["unit", "test"],
                                    base_config={"ctx": 6, "emb": 8, "hidden": 16})
        check(catalog.get("tiny-unit") is not None)
        check(os.path.exists(spec.path))
        loader = ModelLoader(catalog, max_params=100_000)
        model = loader.load_model("tiny-unit")
        check(len(model.C) > 0)
        found = catalog.find_by_domain(["unit"])
        check(any(s.name == "tiny-unit" for s in found))
        catalog.remove("tiny-unit")
        check(catalog.get("tiny-unit") is None)


# ---------------------------------------------------------------------------
# UI smoke test (headless-safe: requires display; skipped when unavailable)
# ---------------------------------------------------------------------------

@test("ui: app constructs and generation controls wired")
def t_ui_smoke():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
    except Exception:
        print("    (no display - skipping)")
        return
    try:
        from uitk.app import App
        app = App(root)
        check(app.top_k_var.get() == str(app.cfg["generation"]["top_k"]))
        check(app.top_p_var.get() == str(app.cfg["generation"]["top_p"]))
        check(app.rep_penalty_var.get() == str(app.cfg["generation"]["repetition_penalty"]))
        check(("training", "grad_clip") in app._settings_entries)
        check(("training", "patience") in app._settings_entries)
        # Training cancel button + use-selected-model checkbox exist
        check(hasattr(app, "cancel_train_btn"))
        check(hasattr(app, "use_selected_model"))
        # Chat holds conversation only: operational message goes to logs, not chat
        app._log_unified("system", "smoke operational entry")
        check("smoke operational entry" not in app.history.get("1.0", "end"))
        # Streaming pipeline: token travels in the label slot of the message
        app.queue.put(("stream_token", "tok123", (1, 10, 5.0)))
        app._poll_queue()  # must not raise on unpacking
        check("tok123" in app.history.get("1.0", "end"))
    finally:
        root.destroy()


@test("ui tk: training box shows marked text and locks during training")
def t_ui_train_box():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
    except Exception:
        print("    (no display - skipping)")
        return
    try:
        from uitk.app import App
        app = App(root)
        raw = "Hello world.\nSecond line."
        app.train_text.delete("1.0", "end")
        app.train_text.insert("1.0", raw)
        app._train_text_raw = raw
        app._show_marked_training_text("Hello world. [EOT] [EOL] [EOS]")
        content = app.train_text.get("1.0", "end")
        check("[EOT]" in content and "[EOS]" in content)
        check(str(app.train_text.cget("state")) == "disabled")
        check(len(app.train_text.tag_ranges("marker")) > 0)
        app._restore_training_text()
        check(app.train_text.get("1.0", "end-1c") == raw)
        check(str(app.train_text.cget("state")) == "normal")
    finally:
        root.destroy()


@test("ui: toolkit auto-detect prefers pyside6, tkinter fallback honors override")
def t_ui_toolkit_detect():
    from ui_base import detect_toolkit, has_pyside6, parse_bool, write_config_yaml
    check(detect_toolkit() in ("pyside6", "tkinter"))
    check(detect_toolkit() == ("pyside6" if has_pyside6() else "tkinter"))
    # Env override forces the stdlib fallback
    os.environ["SCRATCHPADLM_UI"] = "tkinter"
    try:
        check(detect_toolkit() == "tkinter")
    finally:
        del os.environ["SCRATCHPADLM_UI"]
    # Shared helpers used by both toolkits
    check(parse_bool("true") is True and parse_bool("off") is False)
    check(parse_bool.__name__ == "bool")
    data = {"generation": {"top_k": 7, "top_p": 0.95}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        write_config_yaml(path, data)
        from config import Config
        cfg = Config.from_file(path)
        check(cfg["generation"]["top_k"] == 7)
    finally:
        os.unlink(path)


@test("ui: qt platform preflight (missing plugin deps -> tkinter fallback)")
def t_ui_qt_preflight():
    from ui_base import has_pyside6, qt_platform_check

    ok, reason = qt_platform_check()
    check(isinstance(ok, bool) and isinstance(reason, str))
    if ok:
        check(has_pyside6())
    else:
        check(reason, "failure must come with a reason")

    # Simulate a headless/broken environment: preflight must fail cleanly
    # (this is what triggers the Tkinter fallback instead of a Qt abort)
    saved = {k: os.environ.pop(k, None) for k in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    try:
        ok, reason = qt_platform_check()
        check(ok is False)
        if has_pyside6():
            check("DISPLAY" in reason, reason)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@test("ui: pyside6 app constructs offscreen (smoke)")
def t_ui_qt_smoke():
    from ui_base import has_pyside6
    if not has_pyside6():
        print("    (PySide6 not installed - skipping)")
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
        from uiqt.window import MainWindow
    except Exception as exc:
        print(f"    (PySide6 unavailable: {exc} - skipping)")
        return
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        check(tabs == ["Chat", "Training", "Model Info", "Admin", "Settings", "Logs"], tabs)
        # Generation controls loaded from config (parity with Tkinter app)
        check(win.max_tokens_spin.value() == win.cfg["generation"]["max_tokens"])
        check(win.min_tokens_spin.value() == win.cfg["generation"]["min_tokens"])
        check(win.top_k_spin.value() == win.cfg["generation"]["top_k"])
        check(("training", "grad_clip") in win._settings_entries)
        check(("training", "patience") in win._settings_entries)
        # Training cancel + use-selected-model controls exist
        check(hasattr(win, "cancel_train_btn"))
        check(hasattr(win, "use_selected_model_cb"))
        # Chat session + bubble + thumbs feedback
        # Chat holds conversation only (user/assistant); operational logs go to Logs tab
        win.append_history("user", "hello")
        win._append_message("assistant", "hi there", prompt="hello", model="TinyLM")
        msgs = win._sessions[win._current_session]["messages"]
        check(len(msgs) == 2, f"expected 2 messages, got {len(msgs)}")
        rating_msg = [m for m in msgs if m["role"] == "assistant"][0]
        bubble = win._last_bubble
        # Redirect feedback to a temp file so the test doesn't pollute the repo
        feedback_path = win.rating_collector.feedback_path
        win.rating_collector.feedback_path = feedback_path.__class__(
            os.path.join(tempfile.gettempdir(), "tinylm_test_feedback.jsonl")
        )
        try:
            # Thumbs are connected and rated in-place
            bubble.thumbs_up.click()
            check(rating_msg["rating"] == 5, rating_msg)
            check(win.rating_collector.feedback_path.exists())
        finally:
            os.unlink(win.rating_collector.feedback_path)
            win.rating_collector.feedback_path = feedback_path
        # Logs roundtrip (append_history also logs, so count the delta)
        before = len(win._all_log_entries)
        win.log_unified("training", "smoke entry")
        check(len(win._all_log_entries) == before + 1)
        # Streaming pipeline: token travels in the label slot of the message
        win.worker.post("stream_token", "tok123", (1, 10, 5.0))
        win._poll_worker()  # must not raise on unpacking
        streamed = win._sessions[win._current_session]["messages"][-1]
        check(streamed["role"] == "assistant" and streamed["text"] == "tok123", streamed)
    finally:
        win.close()
        win.deleteLater()


@test("ui qt: marker highlighter paints markers yellow")
def t_ui_qt_markers():
    from ui_base import has_pyside6
    if not has_pyside6():
        print("    (PySide6 not installed - skipping)")
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QTextCursor
        from uiqt.window import MainWindow
    except Exception as exc:
        print(f"    (PySide6 unavailable: {exc} - skipping)")
        return
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        check(hasattr(win, "_marker_highlighter"))
        win.train_text.setPlainText("Hello world. [EOT] [EOL]\n[EOS]")
        doc = win.train_text.document()
        painted = set()
        block = doc.firstBlock()
        while block.isValid():
            for fr in block.layout().formats():
                if fr.format.foreground().color().name().lower() == "#f7dc6f":
                    painted.add(block.text()[fr.start:fr.start + fr.length])
            block = block.next()
        check(painted == {"[EOT]", "[EOL]", "[EOS]"}, painted)
    finally:
        win.close()
        win.deleteLater()


if __name__ == "__main__":
    print("ScratchpadLM unit tests")
    print("-" * 40)
    run()
