"""
A fully-featured neural net layer system on top of engine.py's Value autograd.
Designed for text classification (bag-of-words embeddings -> dense layers -> sigmoid).
 
TODO: Add a demo at the bottom that builds a tiny bag-of-words classifier on toy data to show
how it all fits together.
"""

class Module:
    """Base class for all learnable modules. Subclasses must override
    .parameters() to return a flat list of all learnable Value objects."""

    def zero_grad(self):
        """Recursively zero all gradients in this module and its children."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron: computes y = activation( sum_i(w_i * x_i) + b )."""

    def __init__(self, nin, nonlin="relu"):
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(0, label="b")
        self.nonlin = nonlin
        self.nin = nin

    def __call__(self, x):
        """Forward pass: compute activation( sum(w*x) + b )."""
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.nonlin == "relu":
            return act.relu()
        elif self.nonlin == "tanh":
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        act_name = self.nonlin if self.nonlin else "Linear"
        return f"{act_name.upper()}Neuron({self.nin})"


class Layer(Module):
    """A fully-connected layer: nout neurons, each with nin inputs.
    
    Args:
        nin: number of input features
        nout: number of output features (neurons)
        nonlin: activation for all neurons in this layer ('relu', 'tanh', or False).
    """

    def __init__(self, nin, nout, nonlin="relu"):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]
        self.nin, self.nout = nin, nout

    def __call__(self, x):
        """Forward pass.
        
        Args:
            x: list of nin values
        Returns:
            - if nout == 1: a single Value
            - if nout > 1: a list of nout Values
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({self.nin} -> {self.nout}) [{self.neurons[0]}]"


class Embedding(Module):
    """A learnable embedding lookup table, mapping integer token IDs to
    dense vectors.
    
    Imagine you have a vocabulary of 1000 words. Each word gets its own
    learnable 64-dimensional vector. Embedding(1000, 64) gives you a table
    of shape (1000, 64), and embedding[token_id] returns a 64-d vector.
    """

    def __init__(self, vocab_size, embed_dim):
        # Each row is a token's embedding. Initialize small so gradients
        # stay manageable early in training.
        self.table = [
            [
                Value(random.uniform(-0.1, 0.1), label=f"emb[{i},{j}]")
                for j in range(embed_dim)
            ]
            for i in range(vocab_size)
        ]
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, token_ids):
        # Normalize input to always be a list
        if isinstance(token_ids, int):
            token_ids = [token_ids]
            scalar_input = True
        else:
            scalar_input = False

        embeddings = []
        for token_id in token_ids:
            assert (
                0 <= token_id < self.vocab_size
            ), f"token_id {token_id} out of range [0, {self.vocab_size})"
            embeddings.append(self.table[token_id])

        return embeddings[0] if scalar_input else embeddings

    def parameters(self):
        # Flatten the 2D table into a 1D list
        return [v for row in self.table for v in row]

    def __repr__(self):
        return f"Embedding({self.vocab_size} vocab, {self.embed_dim} dim)"


class MLP(Module):
    """Multi-layer perceptron: a stack of fully-connected layers with
    nonlinearities between them.
    
    The output layer is always linear (no activation), so you can apply
    task-specific nonlinearities (sigmoid for binary classification, etc.)
    after the network.
    """

    def __init__(self, nin, nouts, nonlin="relu"):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=nonlin if i != len(nouts) - 1 else False)
            for i in range(len(nouts))
        ]
        self.nin = nin
        self.nouts = nouts

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers_str = " -> ".join(str(sz) for sz in [self.nin] + self.nouts)
        return f"MLP({layers_str})"


class BagOfWordsClassifier(Module):
    """Combines Embedding + MLP for text classification via bag-of-words.
    
    Args:
        vocab_size: number of tokens in the vocabulary
        embed_dim: dimensionality of token embeddings
        hidden_sizes: list of hidden layer sizes in the MLP.
                     E.g. [128, 64] means:
                       embed_dim -> 128 -> 64 -> 1
    """

    def __init__(self, vocab_size, embed_dim, hidden_sizes):
        self.embedding = Embedding(vocab_size, embed_dim)
        # MLP input size is embed_dim (after aggregation)
        self.mlp = MLP(embed_dim, hidden_sizes + [1])

    def __call__(self, token_ids, aggregation="sum"):
        # Look up embeddings
        embeddings = self.embedding(token_ids)  # list of lists

        # Aggregate: sum (or mean) across all tokens
        if aggregation == "sum":
            agg = [
                sum(emb[i] for emb in embeddings) for i in range(self.embedding.embed_dim)
            ]
        elif aggregation == "mean":
            n = len(embeddings)
            agg = [
                sum(emb[i] for emb in embeddings) * (1.0 / n)
                for i in range(self.embedding.embed_dim)
            ]
        else:
            raise ValueError("aggregation must be 'sum' or 'mean'")

        # Pass through MLP
        logit = self.mlp(agg)
        return logit

    def parameters(self):
        return self.embedding.parameters() + self.mlp.parameters()

    def __repr__(self):
        return (
            f"BagOfWordsClassifier(vocab={self.embedding.vocab_size}, "
            f"embed_dim={self.embedding.embed_dim}, mlp={self.mlp})"
        )
