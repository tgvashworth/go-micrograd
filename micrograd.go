package main

import (
	"context"

	"github.com/goccy/go-graphviz"
)

type Graphviz struct {
	G     *graphviz.Graphviz
	Graph *graphviz.Graph
}

func (g *Graphviz) Close() error {
	if err := g.Graph.Close(); err != nil {
		return err
	}
	return g.G.Close()
}

func (g *Graphviz) SetBasicOptions() *Graphviz {
	g.Graph.SetRankDir(graphviz.LRRank)
	g.Graph.SetSize(8, 6)
	g.Graph.SetDPI(600)
	return g
}

func NewGraphviz(ctx context.Context) (Graphviz, error) {
	g, err := graphviz.New(ctx)
	if err != nil {
		return Graphviz{}, err
	}
	graph, err := g.Graph()
	if err != nil {
		return Graphviz{}, err
	}
	self := Graphviz{
		G:     g,
		Graph: graph,
	}
	self.SetBasicOptions()
	return self, nil
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

	a := NewValue(2.0).L("a")
	b := NewValue(-3.0).L("b")
	c := NewValue(10.0).L("c")
	e := a.Mul(b).L("e")
	d := e.Add(c).L("d")
	f := NewValue(-2.0).L("f")
	L := d.Mul(f).L("L")

	trace := TraceFromValue(L)
	trace.DrawGraph(g1.Graph)
	g1.Render(ctx, "basic.png")

	// NEURON
	g2, err := NewGraphviz(ctx)
	if err != nil {
		panic(err)
	}
	defer g2.Close()

	x1 := NewValue(2.0).L("x1")
	x2 := NewValue(0.0).L("x2")
	w1 := NewValue(-3.0).L("w1")
	w2 := NewValue(1.0).L("w2")
	bias := NewValue(6.8813735870195432).L("bias")

	x1w1 := x1.Mul(w1).L("x1w1")
	x2w2 := x2.Mul(w2).L("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).L("x1w1 * 2w2")
	n := x1w1x2w2.Add(bias).L("n")
	o := n.Tanh().L("o").G(1.0)

	// Manual steps of backprop
	// o.back()
	// n.back()
	// x1w1x2w2.back()
	// x1w1.back()
	// x2w2.back()

	// Automatic backprop
	o.ZeroGrad()
	o.Backward()

	trace = TraceFromValue(o)
	trace.DrawGraph(g2.Graph)
	g2.Render(ctx, "neuron.png")
}

func (g *Graphviz) Render(ctx context.Context, filename string) {
	// var dotBuf bytes.Buffer
	// if err := g.Render(ctx, graph, "dot", &dotBuf); err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println(dotBuf.String())

	if err := g.G.RenderFilename(ctx, g.Graph, graphviz.PNG, filename); err != nil {
		panic(err)
	}
}
