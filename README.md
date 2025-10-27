# micrograd

A minimal neural network library built in Go, implementing automatic differentiation and backpropagation from scratch.

## About

This project is a learning exercise to:
- Practice Go programming
- Understand neural networks from first principles
- Implement automatic differentiation (autograd) without frameworks

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

- Scalar-valued autograd engine with automatic backpropagation
- Neural network building blocks: Neuron, Layer, MLP
- Computation graph visualization using Graphviz
- Training example with mean squared error loss

## Usage

```go
// Create a multi-layer perceptron
n := NewMLP(3, []int{4, 4, 1})

// Forward pass
x := []*Value{NewValue(2.0), NewValue(3.0), NewValue(-1.0)}
output := n.Fwd(x)

// Backward pass
loss.Backward()

// Gradient descent
for _, p := range n.Parameters() {
    p.Data += -0.01 * p.Grad
}
```

## Running

```bash
go run .
```
