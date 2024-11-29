// Copyright 2024 The MAE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// Width is the width of the model
	Width = 4
	// Embedding is the embedding size
	Embedding = 2 * Width
	// Factor is the gaussian factor
	Factor = 10000
	// Batch is the batch size
	Batch = 16
	// Networks is the number of networks
	Networks = 3
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

func main() {
	data := Load()
	rng := rand.New(rand.NewSource(1))

	type Network struct {
		Set    tf64.Set
		Others tf64.Set
		L1     tf64.Meta
		L2     tf64.Meta
		Loss   tf64.Meta
		I      int
		V      tf64.Meta
		VV     tf64.Meta
		E      tf64.Meta
	}
	networks := make([]Network, Networks)
	/*drop := 0.9
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}*/
	for n := range networks {
		set := tf64.NewSet()
		set.Add("w1", Embedding, Width/2)
		set.Add("b1", Width/2)
		set.Add("w2", Width/2, Width)
		set.Add("b2", Width)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
		}

		others := tf64.NewSet()
		others.Add("input", Embedding, Batch)
		others.Add("output", Width, Batch)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := /*tf64.Dropout(*/ tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1"))) /*, dropout)*/
		l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
		loss := tf64.Quadratic(l2, others.Get("output"))
		v := tf64.Variance(loss)
		e := tf64.Entropy(tf64.Softmax(loss))
		diff := tf64.Sub(l2, others.Get("output"))
		vv := tf64.Avg(tf64.Variance(tf64.T(tf64.Hadamard(diff, diff))))
		networks[n].Set = set
		networks[n].Others = others
		networks[n].L1 = l1
		networks[n].L2 = l2
		networks[n].Loss = tf64.Avg(loss)
		networks[n].V = v
		networks[n].VV = vv
		networks[n].E = e
	}

	pow := func(x float64, i int) float64 {
		y := math.Pow(x, float64(i+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}

	points := make(plotter.XYs, 0, 8)
	for i := 0; i < 4*33*len(data); i++ {
		factor := Factor*math.Sin(2*math.Pi*float64(i)/256.0) + Factor + .01
		eta := Eta * math.Abs(math.Sin(2*math.Pi*float64(2*i)/256.0))
		index := rng.Intn(len(data))
		network, min := 0, math.MaxFloat64
		for s := 0; s < Batch; s++ {
			transform := MakeRandomTransform(rng, Width, Embedding, factor)
			//offset := MakeRandomTransform(rng, Width, 1, Scale)
			in := NewMatrix(Width, 1, data[index].Measures...)
			in = transform.MulT(in) //.Add(offset).Softmax()
			for n := range networks {
				copy(networks[n].Others.ByName["input"].X[s*Embedding:(s+1)*Embedding], in.Data)
				copy(networks[n].Others.ByName["output"].X[s*Width:(s+1)*Width], data[index].Measures)
			}
		}
		for n := range networks {
			networks[n].Others.Zero()
			networks[n].VV(func(a *tf64.V) bool {
				if a.X[0] < min {
					min, network = a.X[0], n
				}
				return true
			})
		}

		networks[network].Others.Zero()

		networks[network].Set.Zero()
		cost := tf64.Gradient(networks[network].Loss).X[0]

		norm := 0.0
		for _, p := range networks[network].Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1, networks[network].I), pow(B2, networks[network].I)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range networks[network].Set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		networks[network].I++
	}

	histogram := [3][Networks]float64{}
	for shot := 0; shot < 33; shot++ {
		for range data {
			index := rng.Intn(len(data))
			network, min := 0, math.MaxFloat64
			for s := 0; s < Batch; s++ {
				transform := MakeRandomTransform(rng, Width, Embedding, Factor)
				//offset := MakeRandomTransform(rng, Width, 1, Scale)
				in := NewMatrix(Width, 1, data[index].Measures...)
				in = transform.MulT(in) //.Add(offset).Softmax()
				for n := range networks {
					copy(networks[n].Others.ByName["input"].X[s*Embedding:(s+1)*Embedding], in.Data)
					copy(networks[n].Others.ByName["output"].X[s*Width:(s+1)*Width], data[index].Measures)
				}
			}
			for n := range networks {
				networks[n].Others.Zero()
				networks[n].VV(func(a *tf64.V) bool {
					if a.X[0] < min {
						min, network = a.X[0], n
					}
					return true
				})
			}
			data[index].Votes[network]++
		}
	}
	for _, item := range data {
		max, index := 0, 0
		for i, v := range item.Votes {
			if v > max {
				max, index = v, i
			}
		}
		histogram[Labels[item.Label]][index]++
	}
	fmt.Println(histogram)

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}
