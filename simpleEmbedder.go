package simpleEmbedder

import (
	"fmt"
	"reflect"
	"sync"
)

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
	embedding, err := MakeEmbedding(mockEmbedder, []byte("hello, world!"), "hello-world")
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
		embedding, err := MakeEmbedding(mockEmbedder, blob, id)
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
	vector1, _ := MakeEmbedding(hfEmbedder, []byte("George Washington might be the greatest president of them all"), "/page/gw")
	vector2, _ := MakeEmbedding(hfEmbedder, []byte("all work and no play makes jack a dull boy all work and no play makes jack a dull boy all work and..."), "/page/shining")
	vector3, _ := MakeEmbedding(hfEmbedder, []byte("What are we having for supper?"), "/page/supper")
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
