package collection

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	embedders "go-simple-embedding-database/embedders"
	embeddings "go-simple-embedding-database/embeddings"
)

func TestJSONSerializing(t *testing.T) {
	embedder := embedders.Embedder(embedders.MockEmbedder{Id: "mock-embedder"})
	collection := Collection{Id: "test-json-serializing", Embedder: embedder, Embeddings: make(map[string]embeddings.Embedding)}
	JSONBody, err := json.Marshal(collection)
	if err != nil {
		t.Errorf("Could not serialize %v", collection)
	}
	expectedJSONBody := []byte("{\"id\":\"test-json-serializing\"},\"embedder\":{\"id\":\"mock-embedder\"},\"embeddings\":{}}")
	if bytes.Equal(JSONBody, expectedJSONBody) {
		t.Errorf("Unexpected JSON: expected %s, got %s", string(expectedJSONBody), string(JSONBody))
	}

	// TODO: this is failing due to the embedder unmarshaling :/
	//JSONCollection := Collection{}
	//err = json.Unmarshal(JSONBody, &JSONCollection)
	//if err != nil {
	//	t.Errorf("Could not unmarshal %s: %v", string(JSONBody), err)
	//}
}

func TestCollection(t *testing.T) {
	embedder := embedders.Embedder(embedders.MockEmbedder{Id: "mock-embedder"})
	goodEmbedding1, err := embeddings.MakeEmbedding(embedder, []byte("good-embedding-1"), "good-embedding-1")
	goodEmbedding2, err := embeddings.MakeEmbedding(embedder, []byte("good-embedding-2"), "good-embedding-2")

	// create a collection
	if err != nil {
		t.Errorf("Could not create embedding needed for testing: %v", err)
	}
	collection := Collection{Id: "test-collection", Embedder: embedder, Embeddings: make(map[string]embeddings.Embedding)}

	// add two embeddings
	err = collection.AddEmbedding(goodEmbedding1)
	if err != nil {
		t.Errorf("Could not add valid embedding to collection: %v", err)
	}
	err = collection.AddEmbedding(goodEmbedding1)
	if err == nil {
		t.Errorf("Should not have been able to add duplicate embedding %v", goodEmbedding1)
	}
	err = collection.AddEmbedding(goodEmbedding2)
	if err != nil {
		t.Errorf("Could not add valid embedding to collection: %v", err)
	}

	// make sure the two embeddings show up
	if len(collection.Embeddings) != 2 {
		t.Errorf("len(collection.Embeddings) wrong: expected %d, got %d", 2, len(collection.Embeddings))
	}
	result, ok := collection.Embeddings[goodEmbedding1.Id]
	if !ok {
		t.Errorf("Should have been able to find embedding %s", goodEmbedding1.Id)
	}
	if !reflect.DeepEqual(result, *goodEmbedding1) {
		t.Errorf("Embedding %s returned by collection.Embeddings doesn't match the original embedding fed in", goodEmbedding1.Id)
	}
	result, ok = collection.Embeddings[goodEmbedding2.Id]
	if !ok {
		t.Errorf("Should have been able to find embedding %s", goodEmbedding2.Id)
	}
	if !reflect.DeepEqual(result, *goodEmbedding2) {
		t.Errorf("Embedding %s returned by collection.Embeddings doesn't match the original embedding fed in", goodEmbedding2.Id)
	}

	// we shouldn't be able to add a nil embedding
	badEmbedding := embeddings.Embedding{}
	err = collection.AddEmbedding(&badEmbedding)
	if err == nil {
		t.Errorf("Should not have been able to add nil embedding %v to collection", badEmbedding)
	}

	// we shouldn't be able to add an embedding unless the embedders match
	wrongEmbedder := embedders.Embedder(embedders.MockEmbedder{Id: "wrong-embedder"})
	goodEmbeddingWrongEmbedder, err := embeddings.MakeEmbedding(wrongEmbedder, []byte("good-embedding-wrong-embedder"), "good-embedding-wrong-embedder")
	if err != nil {
		t.Errorf("Could not create embedding needed for testing: %v", err)
	}
	err = collection.AddEmbedding(goodEmbeddingWrongEmbedder)
	if err == nil {
		t.Errorf("Should not have been able to add embedding %v to collection - there should be an embedder mismatch", goodEmbeddingWrongEmbedder)
	}

	// we should be able to get the embeddings using collection.GetEmbedding(...)
	embedding, err := collection.GetEmbedding(goodEmbedding1.Id)
	if err != nil {
		t.Errorf("Should have been able to get embedding")
	}
	if !reflect.DeepEqual(embedding, goodEmbedding1) {
		t.Errorf("Embedding %v returned by collection.GetEmbedding(%s) != original embedding %v", embedding, goodEmbedding1.Id, goodEmbedding1)
	}
	embedding, err = collection.GetEmbedding(goodEmbedding2.Id)
	if err != nil {
		t.Errorf("Should have been able to get embedding")
	}
	if !reflect.DeepEqual(embedding, goodEmbedding2) {
		t.Errorf("Embedding %v returned by collection.GetEmbedding(%s) != original embedding %v", embedding, goodEmbedding2.Id, goodEmbedding2)
	}

	// we should be able to delete embeddings just fine too
	err = collection.DeleteEmbedding("does-not-exist")
	if err == nil {
		t.Errorf("Should not have been able to delete non-existent embedding")
	}
	err = collection.DeleteEmbedding(goodEmbedding1.Id)
	if err != nil {
		t.Errorf("Should have been able to delete embedding %s", goodEmbedding1.Id)
	}
	err = collection.DeleteEmbedding(goodEmbedding2.Id)
	if err != nil {
		t.Errorf("Should have been able to delete embedding %s", goodEmbedding2.Id)
	}
	err = collection.DeleteEmbedding(goodEmbedding2.Id)
	if err == nil {
		t.Errorf("Should not have been able to delete embedding %s as it has already been deleted", goodEmbedding2.Id)
	}

	// final check to make sure the collection is clear
	if len(collection.Embeddings) != 0 {
		t.Errorf("Collection should be empty")
	}
}
