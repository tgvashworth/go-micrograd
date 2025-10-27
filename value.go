package main

import (
	"fmt"
	"math"
	"sync/atomic"
)

type Value struct {
	id    uint64
	label string

	Data float64
	Grad float64
	Prev []*Value
	Op   string
}

func (v *Value) GraphLabel() string {
	return fmt.Sprintf("{ %s | d %.4f | g %.4f }", v.Label(), v.Data, v.Grad)
}

var valueCounter uint64

func NewValue(data float64) *Value {
	return NewValueFrom(data, []*Value{}, "")
}

func NewValueFrom(data float64, prev []*Value, op string) *Value {
	id := atomic.AddUint64(&valueCounter, 1)
	return &Value{id: id, Data: data, Prev: prev, Op: op}
}

// DATA

func (v *Value) WithGrad(g float64) *Value {
	v.Grad = g
	return v
}

func (v *Value) G(g float64) *Value {
	return v.WithGrad(g)
}

// LABEL

func (v *Value) Labelled(label string) *Value {
	v.label = label
	return v
}

func (v *Value) L(label string) *Value {
	return v.Labelled(label)
}

func (v *Value) Label() string {
	return v.label
}

// ID + STRING

func (v *Value) ID() string {
	if v.label != "" {
		return fmt.Sprintf("%s_%d", v.label, v.id)
	}

	return fmt.Sprintf("%d", v.id)
}

func (v Value) String() string {
	return fmt.Sprintf("Value(%.4f)", v.Data)
}

// OPERATIONS

func (v *Value) Add(other *Value) *Value {
	return NewValueFrom(v.Data+other.Data, []*Value{v, other}, "+")
}

func (v *Value) Mul(other *Value) *Value {
	return NewValueFrom(v.Data*other.Data, []*Value{v, other}, "*")
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	return NewValueFrom(t, []*Value{v}, "tanh")
}

// BACKPROP

func (v *Value) back() {
	// Rename v to out to help with readability
	out := v
	switch out.Op {
	case "+":
		a, b := out.Prev[0], out.Prev[1]
		a.Grad += 1.0 * out.Grad
		b.Grad += 1.0 * out.Grad
	case "*":
		a, b := out.Prev[0], out.Prev[1]
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	case "tanh":
		p := out.Prev[0]
		t := out.Data
		p.Grad += (1 - t*t) * out.Grad
	default:
		// No operation, do nothing
	}
}

func (v *Value) Backward() {
	visited := make(map[uint64]bool)
	var buildTopo func(n *Value)
	var topo []*Value

	buildTopo = func(n *Value) {
		if !visited[n.id] {
			visited[n.id] = true
			for _, p := range n.Prev {
				buildTopo(p)
			}
			topo = append(topo, n)
		}
	}

	buildTopo(v)

	// Initialize the gradient of the output node
	v.Grad = 1.0

	// Backpropagate in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].back()
	}
}

func (v *Value) ZeroGrad() {
	visited := make(map[uint64]bool)
	var zeroGradRec func(n *Value)

	zeroGradRec = func(n *Value) {
		if !visited[n.id] {
			visited[n.id] = true
			n.Grad = 0.0
			for _, p := range n.Prev {
				zeroGradRec(p)
			}
		}
	}

	zeroGradRec(v)
}
