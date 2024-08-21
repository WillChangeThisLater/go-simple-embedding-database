package utils

import (
	"errors"
	"fmt"
	"math"
)

func CosineSimilarity(x, y []float64) (float64, error) {
	var sum, s1, s2 float64
	if len(x) != len(y) {
		return 0.0, errors.New(fmt.Sprintf("Cannot compare vectors of unequal length (len(x) = %d, len(y) = %d)", len(x), len(y)))
	}
	for i := 0; i < len(x); i++ {
		sum += x[i] * y[i]
		s1 += math.Pow(x[i], 2)
		s2 += math.Pow(y[i], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, nil
	}
	result := sum / (math.Sqrt(s1) * math.Sqrt(s2))
	return result, nil
}
