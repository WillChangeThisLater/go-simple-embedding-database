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
	Id string `json:"id"`
}

func (e MockEmbedder) Embed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil
}

type HuggingFaceEmbedder struct {
	Id      string `json:"id"`
	ModelId string `json:"model_id"`
}

type HuggingFaceOptions struct {
	UseCache     bool `json:"use_cache"`
	WaitForModel bool `json:"wait_for_model"`
}

type HuggingFaceRequestBody struct {
	Inputs []string           `json:"inputs"`
	Value  HuggingFaceOptions `json:"options"`
}

func (e HuggingFaceEmbedder) Embed(blob []byte) ([]float64, error) {
	modelId := e.ModelId

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

	// TODO: I copied this over from another module. Fix it.
	//
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

type Embedder interface {
	Embed(blob []byte) ([]float64, error)
}

type Embedding struct {
	Embedding []float64 `json:"embedding"`
	Embedder  Embedder  `json:"embedder"`
	Blob      []byte    `json:"blob"`
	Id        string    `json:"id"`
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

func makeEmbedding(embedder Embedder, blob []byte, id string) (*Embedding, error) {
	embedding, err := embedder.Embed(blob)
	if err != nil {
		return nil, err
	}
	return &Embedding{Embedding: embedding, Embedder: embedder, Blob: blob, Id: id}, nil
}

func main() {
	mockEmbedder := Embedder(MockEmbedder{Id: "mock-embedder"})

	collection := Collection{Id: "test-collection-api", Embedder: mockEmbedder, Embeddings: make(map[string]Embedding)}
	db := MockDataBase{mutex: &sync.Mutex{}, Collections: make(map[string]Collection)}

	// make sure the collection API works
	err := db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not add collection %v to the database: %v\n", collection, err)
	}
	err = db.AddCollection(&collection)
	if err == nil {
		fmt.Printf("Should not have been able to create collection %v as it already exists\n", collection)
	}
	if !db.isCollectionInDB(collection.Id) {
		fmt.Printf("isCollectionInDB(%s) is reporting collection is not in database when it should be", collection)
	}
	_, err = db.GetCollection(collection.Id)
	if err != nil {
		fmt.Printf("db.getCollection(%s) should have been able to get a collection", collection.Id)
	}

	if len(db.Collections) != 1 {
		fmt.Printf("Database %v should have only 1 collection (has %d)\n", db, len(db.Collections))
	}
	_, ok := db.Collections[collection.Id]
	if !ok {
		fmt.Printf("Collection has the wrong name\n")
	}

	err = db.DeleteCollection(collection.Id)
	if err != nil {
		fmt.Printf("Could not delete collection %s: %v\n", collection, err)
	}
	err = db.DeleteCollection(collection.Id)
	if err == nil {
		fmt.Printf("Should not have been able to delete collection %v as it does not exist\n", collection)
	}
	if db.isCollectionInDB(collection.Id) {
		fmt.Printf("isCollectionInDB(%s) is reporting collection is in database when it shouldn't be", collection)
	}
	_, err = db.GetCollection(collection.Id)
	if err == nil {
		fmt.Printf("db.GetCollection(%s) should have returned an error", collection.Id)
	}

	// now that we know the collection API works, make sure the embedding API works
	collection = Collection{Id: "test-embedding-api-single-vector", Embedder: mockEmbedder, Embeddings: make(map[string]Embedding)}
	embedding, err := makeEmbedding(mockEmbedder, []byte("hello, world!"), "hello-world")
	if err != nil {
		fmt.Printf("Could not manually create embedding: %v\n", err)
	}
	err = db.AddCollection(&collection)
	if err != nil {
		fmt.Printf("Could not add collection %v to the database: %v\n", collection, err)
	}
	err = db.AddEmbedding(collection.Id, embedding)
	if err != nil {
		fmt.Printf("Could not add embedding %v to the database: %v\n", embedding, err)
	}
	err = db.AddEmbedding(collection.Id, &Embedding{Embedding: nil, Embedder: mockEmbedder, Blob: []byte("hey there"), Id: "bad-embedding"})
	if err == nil {
		fmt.Printf("Should not have been able to add embedding with nil embedded values to collection\n")
	}
	//mockEmbedder := Embedder(MockEmbedder{Id: "mock-embedder"})
	err = db.AddEmbedding(collection.Id, &Embedding{Embedding: nil, Embedder: Embedder(MockEmbedder{Id: "mock-embedder-2"}), Blob: []byte("hey there"), Id: "bad-embedding"})
	if err == nil {
		fmt.Printf("Should not have been able to add embedding where collection embedder does not match embedding's embedder\n")
	}
	result, err := db.GetEmbedding(collection.Id, embedding.Id)
	if err != nil {
		fmt.Printf("Could not add embedding %v to the database: %v\n", embedding, err)
	}
	if !reflect.DeepEqual(result, embedding) {
		fmt.Printf("GetEmbedding(%s, %s) returned unexpected embedding (expected %v, got %v)\n", collection.Id, embedding.Id, embedding, result)
	}
	err = db.AddEmbedding(collection.Id, embedding)
	if err == nil {
		fmt.Printf("db.AddEmedding(%s, %v) should not have let me add a duplicate embedding\n", collection.Id, embedding)
	}
	err = db.DeleteEmbedding(collection.Id, embedding.Id)
	if err != nil {
		fmt.Printf("db.DeleteEmbedding(%s, %s) should have let me delete embedding\n", collection.Id, embedding.Id)
	}
	err = db.DeleteEmbedding(collection.Id, embedding.Id)
	if err == nil {
		fmt.Printf("db.DeleteEmbedding(%s, %s) should not have let me delete embedding\n", collection.Id, embedding.Id)
	}
	collection = Collection{Id: "test-embedding-api-many-vectors", Embedder: mockEmbedder, Embeddings: make(map[string]Embedding)}
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
		embedding, err := makeEmbedding(mockEmbedder, blob, id)
		if err != nil {
			fmt.Printf("Could not create embedding: %v", err)
		}
		embeddings = append(embeddings, *embedding)
	}
	for _, embedding := range embeddings {
		db.AddEmbedding(collection.Id, &embedding)
	}
	if len(collection.Embeddings) != embeddingsToGenerate {
		fmt.Printf("Embedding count for collection %s is off (expected %d, got %d)", collection.Id, embeddingsToGenerate, len(collection.Embeddings))
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
		err = db.DeleteEmbedding(collection.Id, embedding.Id)
		if err != nil {
			fmt.Printf("Could not delete embedding %s from collection %s", embedding.Id, collection.Id)
		}
	}

	// great! now let's try embedding something real
	// this should return with no issues...
	hfEmbedder := HuggingFaceEmbedder{Id: "huggingFace", ModelId: "sentence-transformers/all-MiniLM-L12-v2"}
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
	collection = Collection{Id: "test-cosine-similarity", Embedder: hfEmbedder, Embeddings: make(map[string]Embedding)}
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
