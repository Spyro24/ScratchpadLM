"""Minimal scalar autograd (stdlib only). Never send Value objects to subprocesses."""
import math
import operator

# ---- autograd -------------------------------------------------------------

class Value:
    """A scalar that remembers how it was computed, so gradients can flow back."""
    __slots__ = ("data", "grad", "_prev", "_back")

    def __init__(self, data, prev=(), back=None):
        self.data = float(data)
        self.grad = 0.0
        self._prev = prev
        self._back = back          # function that pushes out.grad into the inputs

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def back():
            self.grad += out.grad
            other.grad += out.grad
        out._back = back
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._back = back
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,))

        def back():
            self.grad += n * self.data ** (n - 1) * out.grad
        out._back = back
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,))

        def back():
            self.grad += e * out.grad
        out._back = back
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))

        def back():
            self.grad += out.grad / self.data
        out._back = back
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def back():
            self.grad += (1.0 - t * t) * out.grad
        out._back = back
        return out

    def backward(self):
        # Iterative post-order traversal, then apply each node's backward function
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                topo.append(v)
                continue
            if v in seen:
                continue
            seen.add(v)
            stack.append((v, True))
            for p in v._prev:
                if p not in seen:
                    stack.append((p, False))

        self.grad = 1.0
        for v in reversed(topo):
            if v._back is not None:
                v._back()


def dot(ws, xs, b=None):
    """sum(w * x) + b as a single graph node."""
    wd = [w.data for w in ws]
    xd = [x.data for x in xs]
    total = sum(map(operator.mul, wd, xd))
    prev = tuple(ws) + tuple(xs)
    if b is not None:
        total += b.data
        prev += (b,)
    out = Value(total, prev)

    def back():
        g = out.grad
        for w, xv in zip(ws, xd):
            w.grad += g * xv
        for x, wv in zip(xs, wd):
            x.grad += g * wv
        if b is not None:
            b.grad += g
    out._back = back
    return out


def matvec_dot(rows, xs, biases=None):
    """Compute multiple dot products: [row @ x + b for row, b in zip(rows, biases)].
    More efficient than calling dot() per row - shares forward computation."""
    xd = [x.data for x in xs]
    outs = []
    for i, row in enumerate(rows):
        wd = [w.data for w in row]
        total = sum(map(operator.mul, wd, xd))
        if biases is not None:
            total += biases[i].data
        prev = tuple(row) + tuple(xs)
        if biases is not None:
            prev += (biases[i],)
        out = Value(total, prev)

        # Capture wd and out for this row's backward (avoid late binding)
        def make_back(row=row, wd=wd, b=biases[i] if biases else None, out=out):
            def back():
                g = out.grad
                for w, xv in zip(row, xd):
                    w.grad += g * xv
                for x, wv in zip(xs, wd):
                    x.grad += g * wv
                if b is not None:
                    b.grad += g
            return back
        out._back = make_back()
        outs.append(out)
    return outs


def softmax_cross_entropy(logits, target):
    """Cross-entropy of softmax(logits) against one target index, as one node.
    The gradient with respect to each logit is probs[i] - (1 if i == target)."""
    datas = [l.data for l in logits]
    m = max(datas)
    exps = [math.exp(d - m) for d in datas]
    s = sum(exps)
    loss = -(datas[target] - m - math.log(s))
    probs = [e / s for e in exps]
    out = Value(loss, tuple(logits))

    def back():
        g = out.grad
        for l, p in zip(logits, probs):
            l.grad += g * p
        logits[target].grad -= g
    out._back = back
    return out

