package records

import (
	"errors"
	"testing"

	embedders "go-simple-embedding-database/embedders"
)

type MockEmbedderLongVector struct {
	Id string `json:"id"`
}

func (e MockEmbedderLongVector) Embed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, nil
}

type MockEmbedderShortVector struct {
	Id string `json:"id"`
}

func (e MockEmbedderShortVector) Embed(blob []byte) ([]float64, error) {
	return []float64{1.0}, nil
}

type BadEmbedder struct {
	Id string `json:"id"`
}

func (e BadEmbedder) Embed(blob []byte) ([]float64, error) {
	return nil, errors.New("failure for test")
}

func testMakeRecord(t *testing.T) {
	embedder := embedders.Embedder(BadEmbedder{Id: "bad"})
	_, err := MakeRecord(embedder, []byte("MakeRecord() should not work"), "bad-record")
	if err == nil {
		t.Errorf("Should not have been able to create record")
	}
}

func TestStringer(t *testing.T) {
	// test a short record
	embedder := embedders.Embedder(MockEmbedderShortVector{Id: "mock-embedder-short"})
	blob := "short"
	record, err := MakeRecord(embedder, []byte(blob), "test")
	if err != nil {
		t.Errorf("Should have been able to create record")
	}
	stringValue := record.String()
	expectedValue := "Embedding{Embedding: [1.00], Embedder: {mock-embedder-short}, Blob: short, Id: test}"
	if stringValue != expectedValue {
		t.Errorf("Expected %s, got %s", expectedValue, stringValue)
	}

	// test the ... capabilities
	embedder = embedders.Embedder(MockEmbedderLongVector{Id: "mock-embedder-long"})
	blob = "hey there, this is a long test string. it needs to be over 100 characters long for the ellipses to kick in"
	record, err = MakeRecord(embedder, []byte(blob), "test")
	if err != nil {
		t.Errorf("Should have been able to create record")
	}
	stringValue = record.String()
	expectedValue = "Embedding{Embedding: [1.00, 2.00, 3.00, 4.00, 5.00,  ...], Embedder: {mock-embedder-long}, Blob: hey there, this is a long test string. it needs to be over 100 characters long for the ellipses to k..., Id: test}"
	if stringValue != expectedValue {
		t.Errorf("Expected %s, got %s", expectedValue, stringValue)
	}
}
