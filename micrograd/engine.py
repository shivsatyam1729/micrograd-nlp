"""
A souped-up version of Karpathy's tiny scalar-valued autograd engine
(https://github.com/karpathy/micrograd), tuned for a text-classification
use case: lots of scalar Values (one per token/feature), summed into
long chains, feeding a softmax + cross-entropy head for a 0/1
Kaggle target.

A tiny self-test (numerical gradient checking) lives at the bottom under
`if __name__ == "__main__":`.
"""

class Value:
    """Stores a single scalar value and the subgraph that produced it."""
 
    # Fixed set of instance attributes -> no per-instance __dict__.
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label", "_id")
    _counter = 0  # class-level, gives every Value a unique debug id

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None           # filled in by each op below
        self._prev = set(_children)              # parents in the graph
        self._op = _op                            # op that produced this node
        self.label = label                         # optional, for debugging/graphviz
 
        Value._counter += 1
        self._id = Value._counter
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
 
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
 
        out._backward = _backward
        return out
 
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
 
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
 
        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Two cases now:
          1. Value ** constant   (power rule, same as the original)
          2. Value ** Value      (full f(x)**g(x)):
                d/dx[f**g] = f**g * ( g * f'/f + g' * ln(f) )
             which needs f(x) > 0 for ln(f) to be real, same restriction
             every autograd library has here.
        """
        if isinstance(other, (int, float)):
            out = Value(self.data**other, (self,), f"**{other}")
 
            def _backward():
                self.grad += (other * self.data ** (other - 1)) * out.grad
 
            out._backward = _backward
            return out
 
        if isinstance(other, Value):
            assert self.data > 0, (
                "f(x) ** g(x) needs f(x) > 0 (the gradient needs ln(f(x)))"
            )
            out = Value(self.data**other.data, (self, other), "**")
 
            def _backward():
                self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
                other.grad += out.data * math.log(self.data) * out.grad
 
            out._backward = _backward
            return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")
 
        def _backward():
            self.grad += (out.data > 0) * out.grad
 
        out._backward = _backward
        return out

    def backward(self):
        """
        Topologically sorts the graph feeding into this node, then applies
        the chain rule back-to-front, exactly like the original.
 
        The difference is *how* the topological sort is built. Karpathy's
        recursive build_topo() puts one Python stack frame per edge walked.
        That's invisible for the small expressions in the tutorial, but if
        you build a feature vector like
        """
        topo = []
        visited = {self}
        stack = [(self, iter(self._prev))]
 
        while stack:
            node, children = stack[-1]
            nxt = next(children, None)
            if nxt is None:
                # all children of `node` have been processed -> emit it
                topo.append(node)
                stack.pop()
            elif nxt not in visited:
                visited.add(nxt)
                stack.append((nxt, iter(nxt._prev)))
 
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

        # go one variable at a time and apply the chain rule for the gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def log(self, eps=1e-12):
        """
        Numerically-safe log. A BCE loss will, sooner or later, hand this
        a probability that rounded to exactly 0.0 or 1.0 -- plain
        math.log(0) raises ValueError and kills your training loop.
        Clamp into [eps, 1-eps] first instead.
        """
        if 0.0 <= self.data <= 1.0:
            safe_data = min(max(self.data, eps), 1 - eps)
        else:
            safe_data = max(self.data, eps)  # generic use outside (0,1)
 
        out = Value(math.log(safe_data), (self,), "log")
 
        def _backward():
            self.grad += (1.0 / safe_data) * out.grad
 
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
 
        def _backward():
            self.grad += out.data * out.grad
 
        out._backward = _backward
        return out
 
    def __abs__(self):
        out = Value(abs(self.data), (self,), "abs")
 
        def _backward():
            self.grad += (1.0 if self.data >= 0 else -1.0) * out.grad
 
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
 
    def __radd__(self, other):
        return self + other
 
    def __sub__(self, other):
        return self + (-other)
 
    def __rsub__(self, other):
        return other + (-self)
 
    def __rmul__(self, other):
        return self * other
 
    def __truediv__(self, other):
        return self * other**-1
 
    def __rtruediv__(self, other):
        return other * self**-1
    
    # Comparisons compare on .data only, and deliberately do NOT touch
    # __eq__/__hash__. Value still hashes/compares by identity, which is
    # what the visited-sets in backward()/zero_grad() rely on.
    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Value) else other)
 
    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, Value) else other)
 
    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Value) else other)
 
    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, Value) else other)
 
    def __float__(self):
        return float(self.data)
 
    def __int__(self):
        return int(self.data)
 
    def __repr__(self):
        tag = f", label='{self.label}'" if self.label else ""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}{tag})"
