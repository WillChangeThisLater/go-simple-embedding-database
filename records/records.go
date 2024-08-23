package records

import (
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
