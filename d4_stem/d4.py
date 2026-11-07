"""RESEARCH ONLY (R14 #45): the 8 elements of D4 acting on the last two axes of [...,N,N] arrays, two conventions.
"c64":   reflections about the centred index N//2 used by doe_core.fresnel (x = (n - N//2)*dx): n -> (N - n) mod N,
         implemented as roll(flip(X), 1). Transpose is the same index map in both conventions.
"c63.5": reflections about the pixel-grid midpoint (N-1)/2: plain torch.flip.
Elements are words over (t = transpose, fx = reflect last axis, fy = reflect second-to-last axis), applied right-to-left
as listed: ("t","fx") means t(fx(X))."""
import torch

ELEMENTS = [(), ("fx",), ("fy",), ("fx", "fy"), ("t",), ("t", "fx"), ("t", "fy"), ("t", "fx", "fy")]


def _flip(x, dim, conv):
    y = torch.flip(x, [dim])
    return torch.roll(y, 1, dims=dim) if conv == "c64" else y


def apply(word, x, conv="c64"):
    for op in reversed(word):
        if op == "t":
            x = x.transpose(-1, -2)
        elif op == "fx":
            x = _flip(x, -1, conv)
        elif op == "fy":
            x = _flip(x, -2, conv)
        else:
            raise ValueError(op)
    return x


def inverse(word):
    """Group inverse as a word: each generator is an involution, so reverse the word."""
    return tuple(reversed(word))
