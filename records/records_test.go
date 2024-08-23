package records

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	embedders "go-simple-embedding-database/embedders"
)

func LongEmbed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, nil
}

func ShortEmbed(blob []byte) ([]float64, error) {
	return []float64{1.0}, nil
}

func BadEmbed(blob []byte) ([]float64, error) {
	return nil, errors.New("failure for test")
}

func TestJSON(t *testing.T) {
	embedders.EmbedderRegister["embedder"] = ShortEmbed
	record, err := MakeRecord("embedder", []byte("blob"), "record-id")
	if err != nil {
		t.Errorf("Could not create record: %v", err)
	}

	JSONBody, err := json.Marshal(record)
	expectedJSONBody := []byte("{\"blob\":\"blob\",\"embedding\":[1],\"embedderId\":\"embedder\",\"id\":\"record-id\"}")
	if err != nil {
		t.Errorf("Could not marshal record %v: %v", record, err)
	}
	if !bytes.Equal(JSONBody, expectedJSONBody) {
		t.Errorf("Unexpected JSON: expected %s, got %s", string(expectedJSONBody), string(JSONBody))
	}

	newRecord := &Record{}
	err = json.Unmarshal(JSONBody, newRecord)
	if err != nil {
		t.Errorf("Could not unmarshal JSON: %v", err)
	}
	if !reflect.DeepEqual(record, newRecord) {
		t.Errorf("Unmarshal failed: expected %v, got %v", record, newRecord)
	}
}

func TestMakeRecord(t *testing.T) {
	embedders.EmbedderRegister["mock-bad-embed"] = BadEmbed
	_, err := MakeRecord("mock-bad-embed", []byte("MakeRecord() should not work"), "bad-record")
	if err == nil {
		t.Errorf("Should not have been able to create record")
	}
}

func TestStringer(t *testing.T) {
	// test a short record
	embedders.EmbedderRegister["mock-short-embed"] = ShortEmbed
	blob := "short"
	record, err := MakeRecord("mock-short-embed", []byte(blob), "test")
	if err != nil {
		t.Errorf("Should have been able to create record")
	}
	stringValue := record.String()
	expectedValue := "Embedding{Embedding: [1.00], EmbedderId: mock-short-embed, Blob: short, Id: test}"
	if stringValue != expectedValue {
		t.Errorf("Expected %s, got %s", expectedValue, stringValue)
	}

	// test the ... capabilities
	embedders.EmbedderRegister["mock-long-embed"] = LongEmbed
	blob = "hey there, this is a long test string. it needs to be over 100 characters long for the ellipses to kick in"
	record, err = MakeRecord("mock-long-embed", []byte(blob), "test")
	if err != nil {
		t.Errorf("Should have been able to create record")
	}
	stringValue = record.String()
	expectedValue = "Embedding{Embedding: [1.00, 2.00, 3.00, 4.00, 5.00,  ...], EmbedderId: mock-long-embed, Blob: hey there, this is a long test string. it needs to be over 100 characters long for the ellipses to k..., Id: test}"
	if stringValue != expectedValue {
		t.Errorf("Expected %s, got %s", expectedValue, stringValue)
	}
}
