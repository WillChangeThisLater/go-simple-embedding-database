package records

import (
	"encoding/json"
	"fmt"
	"strconv"

	embedders "go-simple-embedding-database/embedders"
)

type Record struct {
	Embedding  []float64 `json:"embedding"`
	EmbedderId string    `json:"embedderId"`
	Blob       []byte    `json:"blob"`
	Id         string    `json:"id"`
}

// GPT
func (r Record) MarshalJSON() ([]byte, error) {
	// Create a new type alias with the same structure as Record
	type Alias Record

	// Create a new struct with the new type and an additional field for the custom Blob
	newRecord := &struct {
		Blob string `json:"blob"`
		*Alias
	}{
		Blob:  string(r.Blob),
		Alias: (*Alias)(&r),
	}

	return json.Marshal(newRecord)
}

// GPT
func (r *Record) UnmarshalJSON(bytes []byte) error {
	// Create a new type alias with the same structure as Record
	type Alias Record

	// Create a new struct with the new type and an additional field for the custom Blob
	newRecord := &struct {
		Blob string `json:"blob"`
		*Alias
	}{
		Alias: (*Alias)(r),
	}

	if err := json.Unmarshal(bytes, newRecord); err != nil {
		return err
	}

	r.Blob = []byte(newRecord.Blob)
	return nil
}

// TODO: do the ... thing with the embeddings too
func (e Record) String() string {
	defaultBlobLookahead := min(100, len(e.Blob))
	defaultBlob := string(e.Blob[:defaultBlobLookahead])
	if len(e.Blob) > 100 {
		defaultBlob += "..."
	}

	defaultEmbeddingLookahead := min(5, len(e.Embedding))
	precision := 2
	embeddingString := "["
	for _, float := range e.Embedding[:defaultEmbeddingLookahead] {
		embeddingString += strconv.FormatFloat(float, 'f', precision, 64)
		embeddingString += ", "
	}
	if len(e.Embedding) > 5 {
		embeddingString += " ..."
	} else {
		embeddingString = embeddingString[:len(embeddingString)-2]
	}
	embeddingString += "]"

	return fmt.Sprintf("Embedding{Embedding: %s, EmbedderId: %s, Blob: %v, Id: %s}", embeddingString, e.EmbedderId, defaultBlob, e.Id)
}

func MakeRecord(embedderId string, blob []byte, id string) (*Record, error) {
	embed, err := embedders.GetEmbedderFunc(embedderId)
	if err != nil {
		return nil, err
	}

	embedding, err := embed(blob)
	if err != nil {
		return nil, err
	}
	return &Record{Embedding: embedding, EmbedderId: embedderId, Blob: blob, Id: id}, nil
}
