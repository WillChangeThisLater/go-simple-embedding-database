package utils

import "math"

func cosineSimilarity(x, y []float64) float64 {
	var sum, s1, s2 float64
	for i := 0; i < len(x); i++ {
		sum += x[i] * y[i]
		s1 += math.Pow(x[i], 2)
		s2 += math.Pow(y[i], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0
	}
	result := sum / (math.Sqrt(s1) * math.Sqrt(s2))
	return result
}
