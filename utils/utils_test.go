package utils

import (
	"testing"
)

func TestSameVector(t *testing.T) {
	x := []float64{0, 1, 2, 3}
	y := []float64{0, 1, 2, 3}
	similarity, err := CosineSimilarity(x, y)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	expectedSimilarity := 1.0
	if similarity != expectedSimilarity {
		t.Errorf("expected %f, got %f", expectedSimilarity, similarity)
	}
}

func TestOppositeVector(t *testing.T) {
	x := []float64{0, 1, 2, 3}
	y := []float64{0, -1, -2, -3}
	similarity, err := CosineSimilarity(x, y)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	expectedSimilarity := -1.0
	if similarity != expectedSimilarity {
		t.Errorf("expected %f, got %f", expectedSimilarity, similarity)
	}
}

func TestOrthogonalVector(t *testing.T) {
	x := []float64{1, 1}
	y := []float64{1, -1}
	similarity, err := CosineSimilarity(x, y)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	expectedSimilarity := 0.0
	if similarity != expectedSimilarity {
		t.Errorf("expected %f, got %f", expectedSimilarity, similarity)
	}
}

func TestBadComparison(t *testing.T) {
	x := []float64{1, 1}
	y := []float64{1, -1, 2}
	_, err := CosineSimilarity(x, y)
	if err == nil {
		t.Errorf("Should not have been able to compare vectors of unequal length")
	}
}
