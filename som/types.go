package som

import (
	"container/heap"
	"fmt"
	"sort"
)

// float64Slice is a helper type that allows to sort float64 slices using sort.Interface
// It keeps the track of original indexes in a separate idx slice
type float64Slice struct {
	// implements sort.Interface
	sort.Float64Slice
	// keeps track of original indices
	index []int
}

// override Swap method
func (s float64Slice) Swap(i, j int) {
	s.Float64Slice.Swap(i, j)
	s.index[i], s.index[j] = s.index[j], s.index[i]
}

func newFloat64Slice(n ...float64) *float64Slice {
	// we need to avoid overriding original slice
	fSlice := make([]float64, len(n))
	copy(fSlice, n)
	// initialize float slice
	s := &float64Slice{Float64Slice: sort.Float64Slice(fSlice), index: make([]int, len(n))}
	for i := range s.index {
		s.index[i] = i
	}

	return s
}

// float64HeapItem is an item that can be stored in float64Heap
// float64HeapItem implements Stringer interface for easier debugging
type float64Item struct {
	val   float64
	index int
}

func (f float64Item) Val() float64 {
	return f.val
}

func (f float64Item) String() string {
	return fmt.Sprintf("val: %f, index: %d", f.val, f.index)
}

// float64Heap provides an implementation of inverted float64 heap
// inverted heap is a tree structure that has the largest number at its root
// float64Heap it implements heap.Interface
type float64Heap struct {
	// heap nodes
	items []*float64Item
	// heap capacity
	cap int
}

func newFloat64Heap(cap int, items ...*float64Item) (*float64Heap, error) {
	if cap <= 0 {
		return nil, fmt.Errorf("Invalid capacity supplied: %d\n", cap)
	}

	// init the heap
	h := &float64Heap{
		items: items,
		cap:   cap,
	}
	heap.Init(h)

	return h, nil
}

func (h float64Heap) Len() int           { return len(h.items) }
func (h float64Heap) Less(i, j int) bool { return h.items[i].val > h.items[j].val }
func (h float64Heap) Swap(i, j int)      { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *float64Heap) Push(x interface{}) {
	item := x.(*float64Item)
	if len(h.items) < h.cap {
		h.items = append(h.items, item)
		return
	}
	// if we are at full cap, just replace the peak
	if len(h.items) == h.cap && item.val < h.items[0].val {
		h.items[0] = item
		heap.Fix(h, 0)
	}
}

func (h *float64Heap) Pop() interface{} {
	old := *h
	n := len(old.items)
	x := old.items[n-1]
	(*h).items = old.items[0 : n-1]
	return x
}

func (h float64Heap) Cap() int {
	return h.cap
}
