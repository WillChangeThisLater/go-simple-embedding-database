package collection

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	embedders "go-simple-embedding-database/embedders"
	embeddings "go-simple-embedding-database/embeddings"
)

type Collection struct {
	Id         string                          `json:"id"`
	Embedder   embedders.Embedder              `json:"embedder"`
	Embeddings map[string]embeddings.Embedding `json:"embeddings"`
}

// GPT created
func (c *Collection) UnmarshalJSON(data []byte) error {
	// Create a helper struct that mirrors Collection, but with Embedder as a json.RawMessage
	type Alias Collection
	helper := &struct {
		*Alias
		Embedder json.RawMessage `json:"embedder"`
	}{
		Alias: (*Alias)(c),
	}

	// Unmarshal the JSON into the helper struct
	if err := json.Unmarshal(data, &helper); err != nil {
		return err
	}

	var rawEmbedder map[string]interface{}
	if err := json.Unmarshal(helper.Embedder, &rawEmbedder); err != nil {
		return err
	}
	id, ok := rawEmbedder["id"].(string)
	// TODO; write a better error message here.
	if !ok {
		return errors.New("Embedder interface does not have an 'id' field, so it cannot be unmarshaled")
	}

	// Dispatch based on the 'id' field
	switch id {
	case "HuggingFaceEmbedder":
		var embedder embedders.HuggingFaceEmbedder
		if err := json.Unmarshal(helper.Embedder, &embedder); err != nil {
			return err
		}
		c.Embedder = embedder
	case "MockEmbedder":
		var embedder embedders.MockEmbedder
		if err := json.Unmarshal(helper.Embedder, &embedder); err != nil {
			return err
		}
		c.Embedder = embedder
	default:
		return errors.New(fmt.Sprintf("unknown embedder id: %s", id))
	}

	return nil
}

func (c Collection) String() string {
	return fmt.Sprintf("Collection{collection.Id: %s, embedder: %v}", c.Id, c.Embedder)
}

func (collection Collection) AddEmbedding(embedding *embeddings.Embedding) error {
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

func (collection Collection) GetEmbedding(embeddingId string) (*embeddings.Embedding, error) {
	embedding, ok := collection.Embeddings[embeddingId]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Could not get embedding - embedding with ID %s does not exist in collection", embeddingId))
	}
	return &embedding, nil
}
