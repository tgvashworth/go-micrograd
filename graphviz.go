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
	g.Graph.SetSize(16, 12)
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
