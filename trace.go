package main

import (
	"fmt"

	"github.com/goccy/go-graphviz"
)

type Trace struct {
	Nodes []*Value
	Edges [][2]*Value
	Seen  map[*Value]bool
}

func NewTrace() Trace {
	return Trace{
		Nodes: []*Value{},
		Edges: [][2]*Value{},
		Seen:  make(map[*Value]bool),
	}
}

func TraceFromValue(v *Value) Trace {
	t := NewTrace()
	t.build(v)
	return t
}

// Build function that:
//  1. Checks if node has been seen
//  2. If not, mark it as seen, add to nodes list
//  3. For each predecessor
//     a. Add edge
//     b. Recurse
func (t *Trace) build(v *Value) {
	if t.Seen[v] {
		return
	}
	t.Seen[v] = true
	t.Nodes = append(t.Nodes, v)
	for _, child := range v.Prev {
		t.Edges = append(t.Edges, [2]*Value{child, v})
		t.build(child)
	}
}

func (t *Trace) DrawGraph(g *graphviz.Graph) {
	var nodes map[string]*graphviz.Node = make(map[string]*graphviz.Node)
	for _, node := range t.Nodes {
		n, err := g.CreateNodeByName(node.ID())
		if err != nil {
			panic(err)
		}
		n.SetLabel(node.GraphLabel())
		n.SetShape("record")

		if node.Op != "" {
			opName := node.ID() + node.Op
			opN, err := g.CreateNodeByName(opName)
			if err != nil {
				panic(err)
			}
			opN.SetLabel(node.Op)
			_, err = g.CreateEdgeByName(opName, opN, n)
			if err != nil {
				panic(err)
			}
			nodes[opName] = opN
		}

		nodes[node.ID()] = n
	}

	for _, edge := range t.Edges {
		fromID := edge[0].ID()
		toID := edge[1].ID()
		if edge[1].Op != "" {
			toID = toID + edge[1].Op
		}
		fromNode := nodes[fromID]
		toNode := nodes[toID]
		if fromNode == nil || toNode == nil {
			panic(fmt.Sprintf("nil node for edge from %s to %s", fromID, toID))
		}
		_, err := g.CreateEdgeByName(fromID+"_"+toID, fromNode, toNode)
		if err != nil {
			panic(err)
		}
	}
}
