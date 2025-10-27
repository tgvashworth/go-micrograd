package main

import (
	"context"
	"fmt"
	"math/rand"
)

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(nin int) *Neuron {
	weights := make([]*Value, nin)
	for i := range weights {
		weights[i] = NewValue(randFloat(-1, 1))
	}
	bias := NewValue(0.0)
	return &Neuron{Weights: weights, Bias: bias}
}

func randFloat(min, max int) float64 {
	return float64(min) + rand.Float64()*float64(max-min)
}

func (n *Neuron) Fwd(x []*Value) *Value {
	sum := n.Bias
	for i, wx := range n.Weights {
		sum = sum.Add(wx.Mul(x[i]))
	}
	return sum.Tanh()
}

func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, len(n.Weights)+1)
	copy(params, n.Weights)
	params[len(n.Weights)] = n.Bias
	return params
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range neurons {
		neurons[i] = NewNeuron(nin)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Fwd(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Fwd(x)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	layers := make([]*Layer, len(nouts))
	sz := nin
	for i, nout := range nouts {
		layers[i] = NewLayer(sz, nout)
		sz = nout
	}
	return &MLP{Layers: layers}
}

func (m *MLP) Fwd(x []*Value) []*Value {
	out := x
	for _, l := range m.Layers {
		out = l.Fwd(out)
	}
	return out
}

func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

func main() {
	// Initialise Graphviz so we can visualize later
	ctx := context.Background()

	// BASIC
	g1, err := NewGraphviz(ctx)
	if err != nil {
		panic(err)
	}
	defer g1.Close()

	// x1 := NewValue(2.0).L("x1")
	// x2 := NewValue(0.0).L("x2")
	// w1 := NewValue(-3.0).L("w1")
	// w2 := NewValue(1.0).L("w2")
	// bias := NewValue(6.8813735870195432).L("bias")

	// x1w1 := x1.Mul(w1).L("x1w1")
	// x2w2 := x2.Mul(w2).L("x2w2")
	// x1w1x2w2 := x1w1.Add(x2w2).L("x1w1 * 2w2")
	// n := x1w1x2w2.Add(bias).L("n")
	// o := n.Tanh().L("o").G(1.0)

	// Manual steps of backprop
	// o.back()
	// n.back()
	// x1w1x2w2.back()
	// x1w1.back()
	// x2w2.back()

	// Automatic backprop
	// o.ZeroGrad()
	// o.Backward()

	// x := Vs(2.0, 0.0)
	// n := NewMLP(3, []int{4, 4, 1})
	// out := n.Fwd(x)[0]

	// trace := TraceFromValue(out)
	// trace.DrawGraph(g1.Graph)
	// g1.Render(ctx, "network-pre.png")

	// out.ZeroGrad()
	// out.Backward()

	// trace2 := TraceFromValue(out)
	// trace2.DrawGraph(g1.Graph)
	// g1.Render(ctx, "network-post.png")

	RATE := 0.05
	n := NewMLP(3, []int{4, 4, 1})
	fmt.Println("Example (Initial): ", n.Layers[0].Neurons[0].Weights[0], n.Layers[0].Neurons[0].Weights[0].Grad)

	xs := [][]*Value{
		[]*Value(Vs(2.0, 3.0, -1.0)),
		[]*Value(Vs(3.0, -1.0, 0.5)),
		[]*Value(Vs(0.5, 1.0, 1.0)),
		[]*Value(Vs(1.0, 1.0, -1.0)),
	}
	ys := Vs(1.0, -1.0, -1.0, 1.0)
	ypreds := make([]*Value, len(ys))

	GENERATIONS := 100

	var loss *Value
	for j := 0; j < GENERATIONS; j++ {

		// Forward pass to get predictions
		for i, x := range xs {
			ypreds[i] = n.Fwd(x)[0]
		}

		loss = NewValue(0.0)
		for i, ygt := range ys {
			yout := ypreds[i]
			diff := yout.Sub(ygt)
			loss = loss.Add(diff.Pow(2))
		}

		// Backward pass
		// Always zero grads!
		for _, p := range n.Parameters() {
			p.ZeroGrad()
		}

		loss.Backward()

		// Update parameters
		for _, p := range n.Parameters() {
			p.Data += -RATE * p.Grad
		}

		loss = NewValue(0.0)
		for i, ygt := range ys {
			yout := ypreds[i]
			// subtract and square — but we don't have those operations yet!
			diff := yout.Sub(ygt)
			loss = loss.Add(diff.Pow(2))
		}
		fmt.Printf("%d Loss: %.4f\n", j, loss.Data)
	}

	trace := TraceFromValue(loss)
	trace.DrawGraph(g1.Graph)
	g1.Render(ctx, "mlp-loss.png")
}
