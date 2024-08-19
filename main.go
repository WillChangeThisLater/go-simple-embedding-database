package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"reflect"
	"slices"
	"sync"
)

type MockEmbedder struct {
	id string
}

func (e MockEmbedder) Embed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil
}

type HuggingFaceEmbedder struct {
	id      string
	modelId string
}

func (e HuggingFaceEmbedder) Embed(blob []byte) ([]float64, error) {
	modelId := e.modelId

	type HuggingFaceOptions struct {
		UseCache     bool `json:"use_cache"`
		WaitForModel bool `json:"wait_for_model"`
	}

	type HuggingFaceRequestBody struct {
		Inputs []string           `json:"inputs"`
		Value  HuggingFaceOptions `json:"options"`
	}

	apiKey := os.Getenv("HUGGING_FACE_API_KEY")
	if apiKey == "" {
		return nil, errors.New("HUGGING_FACE_API_KEY environment variable not set.")
	}
	endpoint := "https://api-inference.huggingface.co/pipeline/feature-extraction"

	body := HuggingFaceRequestBody{Inputs: []string{string(blob)}, Value: HuggingFaceOptions{UseCache: true, WaitForModel: true}}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/%s", endpoint, modelId)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}

	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	// Check the status code
	//
	// If wait_for_model is set to False and the model is first loading up,
	// it may return a 503 error
	//
	// See https://huggingface.co/docs/api-inference/detailed_parameters
	if resp.StatusCode != 200 {
		io.Copy(os.Stdout, resp.Body)
		panic(fmt.Sprintf("Response has non-200 status code %d. Response body: %v", resp.StatusCode, respBody))
	}

	var embedding [][]float64
	err = json.Unmarshal(respBody, &embedding)
	if err != nil {
		panic(err)
	}

	vector := embedding[0]
	return vector, nil
}

type DataBase interface {
	AddCollection(collection *Collection) error
	DeleteCollection(collectionId string) error
	GetCollection(collectionId string) (*Collection, error)

	AddEmbedding(collectionId string, embedding *Embedding) error
	GetEmbedding(collectionId string, embeddingId string) error
	DeleteEmbedding(collectionId string, embeddingId string) error

	Query(collectionId string, query []byte, n_greatest int) []*Embedding
}

func cosineSimilarity(x, y []float64) float64 {
	var sum, s1, s2 float64
	for i := 0; i < len(x); i++ {
		sum += x[i] * y[i]
		s1 += math.Pow(x[i], 2)
		s2 += math.Pow(y[i], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0
	}
	result := sum / (math.Sqrt(s1) * math.Sqrt(s2))
	return result
}

func (db MockDataBase) Query(collectionId string, query []byte, n_greatest int) (*[]Embedding, error) {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return nil, err
	}

	embedder := collection.embedder
	queryEmbedding, err := embedder.Embed(query)
	if err != nil {
		return nil, err
	}

	if len(collection.embeddings) <= n_greatest {
		embeddings := make([]Embedding, 0)
		for _, embedding := range collection.embeddings {
			embeddings = append(embeddings, embedding)
		}
		return &embeddings, nil
	}

	mostSimilarEmbeddings := make([]Embedding, 0)
	embeddingSimilarities := make(map[string]float64)

	for embeddingId, embedding := range collection.embeddings {
		embeddingSimilarities[embeddingId] = cosineSimilarity(queryEmbedding, embedding.embedding)
	}
	distances := make([]float64, 0)
	for _, distance := range embeddingSimilarities {
		distances = append(distances, distance)
	}
	slices.Sort(distances)
	slices.Reverse(distances)

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
	mutex       *sync.Mutex
	collections map[string]Collection
}

type Embeddings struct {
	embeddings map[string]Embedding
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
	_, ok := db.collections[collection.id]
	if ok {
		err := errors.New(fmt.Sprintf("Cannot create collection %s: a collection with id %s already exists", collection.id, collection.id))
		return err
	} else {
		db.mutex.Lock()
		defer db.mutex.Unlock()
		db.collections[collection.id] = *collection
	}
	return nil
}

func (db MockDataBase) isCollectionInDB(collectionId string) bool {
	collections := db.collections
	_, ok := collections[collectionId]
	return ok
}

func (db MockDataBase) GetCollection(collectionId string) (*Collection, error) {
	collection, ok := db.collections[collectionId]
	if ok {
		return &collection, nil
	}
	return nil, errors.New(fmt.Sprintf("Could not get collection - no collection with ID %s exists in the database", collectionId))
}

func (db MockDataBase) DeleteCollection(collectionId string) error {
	_, ok := db.collections[collectionId]
	if ok {
		db.mutex.Lock()
		defer db.mutex.Unlock()
		delete(db.collections, collectionId)
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
	return db.collections
}

type QueryResponse struct {
	similarity float64
	embedder   Embedder
}

func (e MockEmbedder) Id() string {
	return e.id
}

type Embedder interface {
	Embed(blob []byte) ([]float64, error)
}

type Embedding struct {
	embedding []float64
	embedder  Embedder
	blob      []byte
	id        string
}

func (e Embedding) String() string {
	defaultBlobLookahead := min(100, len(e.blob))
	defaultBlob := string(e.blob[:defaultBlobLookahead])
	if len(e.blob) > 20 {
		defaultBlob += "..."
	}
	return fmt.Sprintf("Embedding{embedding: %f, embedder: %s, blob: %v, id: %s}", e.embedding, e.embedder, defaultBlob, e.id)
}

type Collection struct {
	id         string
	embedder   Embedder
	embeddings map[string]Embedding
}

func (c Collection) String() string {
	return fmt.Sprintf("Collection{collection.id: %s, embedder: %v}", c.id, c.embedder)
}

func (collection Collection) AddEmbedding(embedding *Embedding) error {
	_, ok := collection.embeddings[embedding.id]
	if ok {
		return errors.New(fmt.Sprintf("Embedding %s already exists in collection %s\n", embedding.id, collection.id))
	}
	collection.embeddings[embedding.id] = *embedding
	return nil
}

func (collection Collection) DeleteEmbedding(embeddingId string) error {
	_, ok := collection.embeddings[embeddingId]
	if ok {
		delete(collection.embeddings, embeddingId)
		return nil
	}
	return errors.New(fmt.Sprintf("Could not delete embedding %s from collection %s: embedding not found in collection", embeddingId, collection.id))
}

func (collection Collection) GetEmbedding(embeddingId string) (*Embedding, error) {
	embedding, ok := collection.embeddings[embeddingId]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Could not get embedding - embedding with ID %s does not exist in collection", embeddingId))
	}
	return &embedding, nil
}

func makeEmbedding(embedder Embedder, blob []byte, id string) (*Embedding, error) {
	embedding, err := embedder.Embed(blob)
	if err != nil {
		return nil, err
	}
	return &Embedding{embedding: embedding, embedder: embedder, blob: blob, id: id}, nil
}

func main() {
	embedder := Embedder(MockEmbedder{id: "mock-embedder"})

	collection := Collection{id: "test-collection-api", embedder: embedder, embeddings: make(map[string]Embedding)}
	db := MockDataBase{mutex: &sync.Mutex{}, collections: make(map[string]Collection)}

	// make sure the collection API works
	err := db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not add collection %v to the database: %v\n", collection, err)
	}
	err = db.AddCollection(&collection)
	if err == nil {
		fmt.Printf("Should not have been able to create collection %v as it already exists\n", collection)
	}
	if !db.isCollectionInDB(collection.id) {
		fmt.Printf("isCollectionInDB(%s) is reporting collection is not in database when it should be", collection)
	}
	_, err = db.GetCollection(collection.id)
	if err != nil {
		fmt.Printf("db.getCollection(%s) should have been able to get a collection", collection.id)
	}

	if len(db.collections) != 1 {
		fmt.Printf("Database %v should have only 1 collection (has %d)\n", db, len(db.collections))
	}
	_, ok := db.collections[collection.id]
	if !ok {
		fmt.Printf("Collection has the wrong name\n")
	}

	err = db.DeleteCollection(collection.id)
	if err != nil {
		fmt.Printf("Could not delete collection %s: %v\n", collection, err)
	}
	err = db.DeleteCollection(collection.id)
	if err == nil {
		fmt.Printf("Should not have been able to delete collection %v as it does not exist\n", collection)
	}
	if db.isCollectionInDB(collection.id) {
		fmt.Printf("isCollectionInDB(%s) is reporting collection is in database when it shouldn't be", collection)
	}
	_, err = db.GetCollection(collection.id)
	if err == nil {
		fmt.Printf("db.GetCollection(%s) should have returned an error", collection.id)
	}

	// now that we know the collection API works, make sure the embedding API works
	collection = Collection{id: "test-embedding-api-single-vector", embedder: embedder, embeddings: make(map[string]Embedding)}
	embedding, err := makeEmbedding(embedder, []byte("hello, world!"), "hello-world")
	if err != nil {
		fmt.Printf("Could not manually create embedding: %v\n", err)
	}
	err = db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not add collection %v to the database: %v\n", collection, err)
	}
	err = db.AddEmbedding(collection.id, embedding)
	if err != nil {
		fmt.Printf("Could not add embedding %v to the database: %v\n", embedding, err)
	}
	result, err := db.GetEmbedding(collection.id, embedding.id)
	if err != nil {
		fmt.Printf("Could not add embedding %v to the database: %v\n", embedding, err)
	}
	if !reflect.DeepEqual(result, embedding) {
		fmt.Printf("GetEmbedding(%s, %s) returned unexpected embedding (expected %v, got %v)\n", collection.id, embedding.id, embedding, result)
	}
	err = db.AddEmbedding(collection.id, embedding)
	if err == nil {
		fmt.Printf("db.AddEmedding(%s, %v) should not have let me add a duplicate embedding\n", collection.id, embedding)
	}
	err = db.DeleteEmbedding(collection.id, embedding.id)
	if err != nil {
		fmt.Printf("db.DeleteEmbedding(%s, %s) should have let me delete embedding\n", collection.id, embedding.id)
	}
	err = db.DeleteEmbedding(collection.id, embedding.id)
	if err == nil {
		fmt.Printf("db.DeleteEmbedding(%s, %s) should not have let me delete embedding\n", collection.id, embedding.id)
	}
	collection = Collection{id: "test-embedding-api-many-vectors", embedder: embedder, embeddings: make(map[string]Embedding)}
	err = db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not add collection %v to the database: %v\n", collection, err)
	}

	// now that we know the collection API works against one embedding, let's test it with multiple
	embeddings := make([]Embedding, 0)
	embeddingsToGenerate := 50
	for pageNum := range embeddingsToGenerate {
		blob := []byte(fmt.Sprintf("Content for page %d\n", pageNum))
		id := fmt.Sprintf("/page/%d", pageNum)
		embedding, err := makeEmbedding(embedder, blob, id)
		if err != nil {
			fmt.Printf("Could not create embedding: %v", err)
		}
		embeddings = append(embeddings, *embedding)
	}
	for _, embedding := range embeddings {
		db.AddEmbedding(collection.id, &embedding)
	}
	if len(collection.embeddings) != embeddingsToGenerate {
		fmt.Printf("Embedding count for collection %s is off (expected %d, got %d)", collection.id, embeddingsToGenerate, len(collection.embeddings))
	}

	// now that we have a bunch of embeddings in the database, let's check the query methods
	n_greatest := 5
	response, err := db.Query("test-embedding-api-many-vectors", []byte("hey hey!"), n_greatest)
	if err != nil {
		fmt.Printf("Query method failed: %v\n", err)
	}
	if len(*response) != n_greatest {
		fmt.Printf("len(*response) != n_greatest (%d != %d)n", len(*response), n_greatest)
	}

	for _, embedding := range embeddings {
		err = db.DeleteEmbedding(collection.id, embedding.id)
		if err != nil {
			fmt.Printf("Could not delete embedding %s from collection %s", embedding.id, collection.id)
		}
	}

	// great! now let's try embedding something real
	// this should return with no issues...
	hfEmbedder := HuggingFaceEmbedder{id: "huggingFace", modelId: "sentence-transformers/all-MiniLM-L12-v2"}
	_, err = hfEmbedder.Embed([]byte("George Washington was the greatest president of them all"))
	if err != nil {
		fmt.Printf("Hugging face embedder could not embed blob: %v", err)
	}

	// now for the real test: does the embedder functionality work under a semi-real scenario?
	// the idea here is that we make embeddings for 3 vastly different sentences, then check the
	// query results using n_greatest=1 to ensure the correct sentence is returned (given a
	// relevant query)
	//
	// this is something of a bad test. it relies on the embedder returning reasonable results.
	// however, i think it's also reasonable to expect that a semi-decent embedding model
	// will be able to embed these sentences somewhat appropriately. so i'm leaving the test in
	vector1, _ := makeEmbedding(hfEmbedder, []byte("George Washington might be the greatest president of them all"), "/page/gw")
	vector2, _ := makeEmbedding(hfEmbedder, []byte("all work and no play makes jack a dull boy all work and no play makes jack a dull boy all work and..."), "/page/shining")
	vector3, _ := makeEmbedding(hfEmbedder, []byte("What are we having for supper?"), "/page/supper")
	collection = Collection{id: "test-cosine-similarity", embedder: hfEmbedder, embeddings: make(map[string]Embedding)}
	err = db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not create collection: %v", err)
	}

	collection.AddEmbedding(vector1)
	collection.AddEmbedding(vector2)
	collection.AddEmbedding(vector3)
	queryResult, err := db.Query("test-cosine-similarity", []byte("Abraham Lincoln, Thomas Jefferson, John F Kennedy"), 1)
	if err != nil {
		fmt.Printf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*vector1, (*queryResult)[0]) {
		fmt.Printf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}

	queryResult, err = db.Query("test-cosine-similarity", []byte("The Shining"), 1)
	if err != nil {
		fmt.Printf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*vector2, (*queryResult)[0]) {
		fmt.Printf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}

	queryResult, err = db.Query("test-cosine-similarity", []byte("We are having chicken and rice for supper, with a side of salad"), 1)
	if err != nil {
		fmt.Printf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*vector3, (*queryResult)[0]) {
		fmt.Printf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}
}
