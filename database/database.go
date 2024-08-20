package database

import (
	"errors"
	"fmt"
	"slices"
	"sync"
)

type DataBase interface {
	AddCollection(collection *Collection) error
	DeleteCollection(collectionId string) error
	GetCollection(collectionId string) (*Collection, error)

	AddEmbedding(collectionId string, embedding *Embedding) error
	GetEmbedding(collectionId string, embeddingId string) error
	DeleteEmbedding(collectionId string, embeddingId string) error

	Query(collectionId string, query []byte, n_greatest int) []*Embedding
}

// TODO: if there's a way to refactor this, do it. It's incredibly ugly.
// Specifically, I don't have a good way to pluck the max N elements
// from a list
func (db MockDataBase) Query(collectionId string, query []byte, n_greatest int) (*[]Embedding, error) {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return nil, err
	}

	embedder := collection.Embedder
	queryEmbedding, err := embedder.Embed(query)
	if err != nil {
		return nil, err
	}

	if len(collection.Embeddings) <= n_greatest {
		embeddings := make([]Embedding, 0)
		for _, embedding := range collection.Embeddings {
			embeddings = append(embeddings, embedding)
		}
		return &embeddings, nil
	}

	mostSimilarEmbeddings := make([]Embedding, 0)
	embeddingSimilarities := make(map[string]float64)

	for embeddingId, embedding := range collection.Embeddings {
		embeddingSimilarities[embeddingId] = cosineSimilarity(queryEmbedding, embedding.Embedding)
	}
	distances := make([]float64, 0)
	for _, distance := range embeddingSimilarities {
		distances = append(distances, distance)
	}
	slices.Sort(distances)
	slices.Reverse(distances)

	// this is an ugly hack to deal with potential duplicate values
	nthGreatestSimilarity := distances[n_greatest-1]
	if nthGreatestSimilarity == distances[n_greatest] {
		numPicked := 0
		for embeddingId, distance := range embeddingSimilarities {
			if distance > nthGreatestSimilarity {
				embedding, err := collection.GetEmbedding(embeddingId)
				if err != nil {
					return nil, err
				}
				mostSimilarEmbeddings = append(mostSimilarEmbeddings, *embedding)
				numPicked += 1
			}
		}
		for embeddingId, distance := range embeddingSimilarities {
			if distance == nthGreatestSimilarity {
				embedding, err := collection.GetEmbedding(embeddingId)
				if err != nil {
					return nil, err
				}
				mostSimilarEmbeddings = append(mostSimilarEmbeddings, *embedding)
				numPicked += 1
				if numPicked == n_greatest {
					if len(mostSimilarEmbeddings) != n_greatest {
						return nil, errors.New(fmt.Sprintf("matching - len(mostSimilarEmbeddings) != n_greatest (%d != %d)", len(mostSimilarEmbeddings), n_greatest))
					}
					return &mostSimilarEmbeddings, nil
				}
			}
		}
	}

	// even the straightforward case is a little ugly
	for embeddingId, distance := range embeddingSimilarities {
		if distance >= nthGreatestSimilarity {
			embedding, err := collection.GetEmbedding(embeddingId)
			if err != nil {
				return nil, err
			}
			mostSimilarEmbeddings = append(mostSimilarEmbeddings, *embedding)
		}
	}
	if len(mostSimilarEmbeddings) != n_greatest {
		return nil, errors.New(fmt.Sprintf("distinct - len(mostSimilarEmbeddings) != n_greatest (%d != %d)", len(mostSimilarEmbeddings), n_greatest))
	}
	return &mostSimilarEmbeddings, nil
}

type MockDataBase struct {
	mutex       *sync.Mutex           // TODO: leverage this
	Collections map[string]Collection `json:"collections"`
}

func (db MockDataBase) AddEmbedding(collectionId string, embedding *Embedding) error {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return err
	}
	return collection.AddEmbedding(embedding)
}

func (db MockDataBase) GetEmbedding(collectionId string, embeddingId string) (*Embedding, error) {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return nil, err
	}
	return collection.GetEmbedding(embeddingId)
}

func (db MockDataBase) DeleteEmbedding(collectionId string, embeddingId string) error {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return err
	}
	return collection.DeleteEmbedding(embeddingId)
}

func (db MockDataBase) AddCollection(collection *Collection) error {
	_, ok := db.Collections[collection.Id]
	if ok {
		err := errors.New(fmt.Sprintf("Cannot create collection %s: a collection with id %s already exists", collection.Id, collection.Id))
		return err
	} else {
		db.mutex.Lock()
		defer db.mutex.Unlock()
		db.Collections[collection.Id] = *collection
	}
	return nil
}

func (db MockDataBase) isCollectionInDB(collectionId string) bool {
	collections := db.Collections
	_, ok := collections[collectionId]
	return ok
}

func (db MockDataBase) GetCollection(collectionId string) (*Collection, error) {
	collection, ok := db.Collections[collectionId]
	if ok {
		return &collection, nil
	}
	return nil, errors.New(fmt.Sprintf("Could not get collection - no collection with ID %s exists in the database", collectionId))
}

func (db MockDataBase) DeleteCollection(collectionId string) error {
	_, ok := db.Collections[collectionId]
	if ok {
		db.mutex.Lock()
		defer db.mutex.Unlock()
		delete(db.Collections, collectionId)
	} else {
		err := errors.New(fmt.Sprintf("Cannot delete collection %s: does not exist", collectionId))
		return err
	}
	return nil
}

func (db MockDataBase) GetCollections() map[string]Collection {
	// I think the locking here is needed?
	db.mutex.Lock()
	defer db.mutex.Unlock()
	return db.Collections
}
