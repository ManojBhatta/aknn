class Value:
    def __init__(self, value, children=(), op=""):
        self.value = value
        self.grad = 0

        # variables for graph construction
        self._backward = lambda: None
        self._op = op  # operation that produced this node
        self._prev = set(
            children
        )  # set of children used to create this, actually could be called parents

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.value + other.value, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self): # -self
        return self * -1

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __sub__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.value - other.value, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.value * other.value, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Value(self.value**other, (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * (other * self.value ** (other - 1))

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.value), (self,), f"relu")

        def _backward():
            g = 0 if self.value < 0 else 1
            self.grad += out.grad * g

        out._backward = _backward
        return out

    def backward(self):
        # a list of child notdes in topological order
        topo = []

        # a set to keep track if a child is already added
        visited = set()

        # built the set in topological order
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1

        for v in reversed(topo):
            v._backward()
