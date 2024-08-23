package embedders

import (
	"testing"
)

func MockEmbed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil
}

func TestEmbedders(t *testing.T) {
	_, err := GetEmbedderFunc("not-registered")
	if err == nil {
		t.Errorf("Should not have been able to get embedder func")
	}

	EmbedderRegister["mock-embedder"] = MockEmbed
	_, err = GetEmbedderFunc("mock-embedder")
	if err != nil {
		t.Errorf("Could not get embedder: %v", err)
	}
}
