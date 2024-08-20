package embeddings

import (
	"fmt"

	embedders "go-simple-embedding-database/embedders"
)

type Embedding struct {
	Embedding []float64          `json:"embedding"`
	Embedder  embedders.Embedder `json:"embedder"`
	Blob      []byte             `json:"blob"`
	Id        string             `json:"id"`
}

// TODO: do the ... thing with the embeddings too
func (e Embedding) String() string {
	defaultBlobLookahead := min(100, len(e.Blob))
	defaultBlob := string(e.Blob[:defaultBlobLookahead])
	if len(e.Blob) > 20 {
		defaultBlob += "..."
	}
	return fmt.Sprintf("Embedding{embedding: %f, embedder: %s, blob: %v, id: %s}", e.Embedding, e.Embedder, defaultBlob, e.Id)
}

func MakeEmbedding(embedder embedders.Embedder, blob []byte, id string) (*Embedding, error) {
	embedding, err := embedder.Embed(blob)
	if err != nil {
		return nil, err
	}
	return &Embedding{Embedding: embedding, Embedder: embedder, Blob: blob, Id: id}, nil
}
