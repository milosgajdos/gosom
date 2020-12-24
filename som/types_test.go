package som

import (
	"container/heap"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewHeap(t *testing.T) {
	assert := assert.New(t)

	highest := 9.0
	fItems := []*float64Item{
		{val: highest, index: 2},
		{val: 7.0, index: 4},
		{val: 8.0, index: 0}}
	// can't have negative capacity
	cap := -4
	errString := "invalid capacity supplied: %d"
	h, err := newFloat64Heap(cap)
	assert.Nil(h)
	assert.EqualError(err, fmt.Sprintf(errString, cap))
	// init map and push items in
	cap = 4
	h, err = newFloat64Heap(cap)
	assert.NotNil(h)
	assert.NoError(err)
	assert.Equal(cap, h.Cap())
	for _, fItem := range fItems {
		heap.Push(h, fItem)
	}
	assert.Equal(len(fItems), h.Len())
	// if we pop, heap length should decrease
	item := heap.Pop(h)
	assert.Equal(len(fItems)-1, h.Len())
	assert.Equal(highest, item.(*float64Item).Val())
	// push large value and check the lowest -- we have popped 8.0 earlier
	highest = 10.0
	heap.Push(h, &float64Item{val: highest, index: 5})
	item = heap.Pop(h)
	assert.Equal(highest, item.(*float64Item).Val())
	// push more items than capacity
	cap = 4
	h, err = newFloat64Heap(cap)
	assert.NotNil(h)
	assert.NoError(err)
	// test float items
	fItems = []*float64Item{
		{val: 9.0, index: 1},
		{val: 7.0, index: 4},
		{val: 9.0, index: 2},
		{val: 8.0, index: 0},
		{val: 9.0, index: 3}}
	for _, fItem := range fItems {
		heap.Push(h, fItem)
	}
	assert.Equal(cap, h.Len())
	item = heap.Pop(h)
	assert.Equal(9.0, item.(*float64Item).Val())
}
