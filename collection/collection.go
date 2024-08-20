package collection

import (
	"errors"
	"fmt"
	"reflect"
)

type Collection struct {
	Id         string               `json:"id"`
	Embedder   Embedder             `json:"embedder"`
	Embeddings map[string]Embedding `json:"embeddings"`
}

func (c Collection) String() string {
	return fmt.Sprintf("Collection{collection.Id: %s, embedder: %v}", c.Id, c.Embedder)
}

func (collection Collection) AddEmbedding(embedding *Embedding) error {
	_, ok := collection.Embeddings[embedding.Id]
	if ok {
		return errors.New(fmt.Sprintf("Embedding %s already exists in collection %s\n", embedding.Id, collection.Id))
	}
	if !reflect.DeepEqual(collection.Embedder, embedding.Embedder) {
		return errors.New(fmt.Sprintf("Embedding embedder %v != collection embedder %v", embedding.Embedder, collection.Embedder))
	}
	if embedding.Embedding == nil {
		return errors.New(fmt.Sprintf("Embedding for %v is null", embedding))
	}
	collection.Embeddings[embedding.Id] = *embedding
	return nil
}

func (collection Collection) DeleteEmbedding(embeddingId string) error {
	_, ok := collection.Embeddings[embeddingId]
	if ok {
		delete(collection.Embeddings, embeddingId)
		return nil
	}
	return errors.New(fmt.Sprintf("Could not delete embedding %s from collection %s: embedding not found in collection", embeddingId, collection.Id))
}

func (collection Collection) GetEmbedding(embeddingId string) (*Embedding, error) {
	embedding, ok := collection.Embeddings[embeddingId]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Could not get embedding - embedding with ID %s does not exist in collection", embeddingId))
	}
	return &embedding, nil
}
