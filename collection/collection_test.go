package collection

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	embedders "go-simple-embedding-database/embedders"
	records "go-simple-embedding-database/records"
)

type MockEmbedder struct {
	Id string `json:"id"`
}

func (e MockEmbedder) Embed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil
}

func TestJSONSerializing(t *testing.T) {
	embedder := embedders.Embedder(MockEmbedder{Id: "mock-embedder"})
	collection := Collection{Id: "test-json-serializing", Embedder: embedder, Records: make(map[string]records.Record)}
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
	embedder := embedders.Embedder(MockEmbedder{Id: "mock-embedder"})
	goodRecord1, err := records.MakeRecord(embedder, []byte("good-record-1"), "good-record-1")
	goodRecord2, err := records.MakeRecord(embedder, []byte("good-record-2"), "good-record-2")

	// create a collection
	if err != nil {
		t.Errorf("Could not create records needed for testing: %v", err)
	}
	collection := Collection{Id: "test-collection", Embedder: embedder, Records: make(map[string]records.Record)}

	// add two embeddings
	err = collection.AddRecord(goodRecord1)
	if err != nil {
		t.Errorf("Could not add valid record to collection: %v", err)
	}
	err = collection.AddRecord(goodRecord1)
	if err == nil {
		t.Errorf("Should not have been able to add duplicate record %v", goodRecord1)
	}
	err = collection.AddRecord(goodRecord2)
	if err != nil {
		t.Errorf("Could not add valid record to collection: %v", err)
	}

	// make sure the two embeddings show up
	if len(collection.Records) != 2 {
		t.Errorf("len(collection.Records) wrong: expected %d, got %d", 2, len(collection.Records))
	}
	result, ok := collection.Records[goodRecord1.Id]
	if !ok {
		t.Errorf("Should have been able to find record %s", goodRecord1.Id)
	}
	if !reflect.DeepEqual(result, *goodRecord1) {
		t.Errorf("Embedding %s returned by collection.Records doesn't match the original record fed in", goodRecord1.Id)
	}
	result, ok = collection.Records[goodRecord2.Id]
	if !ok {
		t.Errorf("Should have been able to find record %s", goodRecord2.Id)
	}
	if !reflect.DeepEqual(result, *goodRecord2) {
		t.Errorf("Embedding %s returned by collection.Records doesn't match the original record fed in", goodRecord2.Id)
	}

	// we shouldn't be able to add a nil embedding
	nilRecord := records.Record{}
	err = collection.AddRecord(&nilRecord)
	if err == nil {
		t.Errorf("Should not have been able to add nil record %v to collection", nilRecord)
	}

	// we shouldn't be able to add an record unless the embedders match
	wrongEmbedder := embedders.Embedder(MockEmbedder{Id: "wrong-embedder"})
	goodRecordWrongEmbedder, err := records.MakeRecord(wrongEmbedder, []byte("good-embedding-wrong-embedder"), "good-embedding-wrong-embedder")
	if err != nil {
		t.Errorf("Could not create record needed for testing: %v", err)
	}
	err = collection.AddRecord(goodRecordWrongEmbedder)
	if err == nil {
		t.Errorf("Should not have been able to add record %v to collection - there should be an embedder mismatch", goodRecordWrongEmbedder)
	}

	// we should be able to get the embeddings using collection.GetRecord(...)
	embedding, err := collection.GetRecord(goodRecord1.Id)
	if err != nil {
		t.Errorf("Should have been able to get embedding")
	}
	if !reflect.DeepEqual(embedding, goodRecord1) {
		t.Errorf("Embedding %v returned by collection.GetRecord(%s) != original record %v", embedding, goodRecord1.Id, goodRecord1)
	}
	embedding, err = collection.GetRecord(goodRecord2.Id)
	if err != nil {
		t.Errorf("Should have been able to get embedding")
	}
	if !reflect.DeepEqual(embedding, goodRecord2) {
		t.Errorf("Embedding %v returned by collection.GetRecord(%s) != original record %v", embedding, goodRecord2.Id, goodRecord2)
	}

	// we should be able to delete embeddings just fine too
	err = collection.DeleteRecord("does-not-exist")
	if err == nil {
		t.Errorf("Should not have been able to delete non-existent record")
	}
	err = collection.DeleteRecord(goodRecord1.Id)
	if err != nil {
		t.Errorf("Should have been able to delete record %s", goodRecord1.Id)
	}
	err = collection.DeleteRecord(goodRecord2.Id)
	if err != nil {
		t.Errorf("Should have been able to delete record %s", goodRecord2.Id)
	}
	err = collection.DeleteRecord(goodRecord2.Id)
	if err == nil {
		t.Errorf("Should not have been able to delete record %s as it has already been deleted", goodRecord2.Id)
	}

	// final check to make sure the collection is clear
	if len(collection.Records) != 0 {
		t.Errorf("Collection should be empty")
	}
}

func TestQueryAgainstContrivedEmbeddings(t *testing.T) {
	embedder := embedders.Embedder(MockEmbedder{Id: "mock-embedder"})
	collection := Collection{Id: "test-many-embeddings", Embedder: embedder, Records: make(map[string]records.Record)}

	newRecords := make([]records.Record, 0)
	recordsToGenerate := 50
	for pageNum := range recordsToGenerate {
		blob := []byte(fmt.Sprintf("Content for page %d\n", pageNum))
		id := fmt.Sprintf("/page/%d", pageNum)
		record, err := records.MakeRecord(embedder, blob, id)
		if err != nil {
			t.Errorf("Could not create record: %v", err)
		}
		newRecords = append(newRecords, *record)
	}
	for _, record := range newRecords {
		collection.AddRecord(&record)
	}
	if len(collection.Records) != recordsToGenerate {
		t.Errorf("Record count for collection %s is off (expected %d, got %d)", collection.Id, recordsToGenerate, len(collection.Records))
	}

	// now that we have a bunch of embeddings in the database, let's check also the query methods
	n_greatest := 5
	response, err := collection.Query([]byte("hey hey!"), n_greatest)
	if err != nil {
		t.Errorf("Query method failed: %v\n", err)
	}
	if len(*response) != n_greatest {
		t.Errorf("len(*response) != n_greatest (%d != %d)n", len(*response), n_greatest)
	}

	for _, record := range newRecords {
		err = collection.DeleteRecord(record.Id)
		if err != nil {
			t.Errorf("Could not delete record %s from collection %s", record.Id, collection.Id)
		}
	}
	if len(collection.Records) != 0 {
		t.Errorf("Collection %s should be empty but records still remain", collection.Id)
	}
}

func TestQueryAgainstRealEmbeddings(t *testing.T) {
	// does the embedder functionality work under a semi-real scenario?
	// the idea here is that we make embeddings for 3 vastly different sentences, then check the
	// query results using n_greatest=1 to ensure the correct sentence is returned (given a
	// relevant query)
	//
	// this is something of a bad test. it relies on the hugging face API, and assumes
	// that the embedder we are calling returns reasonable results.
	// however, i think it's also reasonable to expect that a semi-decent embedding model
	// will be able to embed these sentences somewhat appropriately. so i'm leaving the test in
	hfEmbedder := embedders.HuggingFaceEmbedder{Id: "huggingFace", ModelId: "sentence-transformers/all-MiniLM-L12-v2"}
	_, err := hfEmbedder.Embed([]byte("George Washington was the greatest president of them all"))
	if err != nil {
		t.Errorf("Hugging face embedder could not embed blob: %v", err)
	}

	record1, _ := records.MakeRecord(hfEmbedder, []byte("George Washington might be the greatest president of them all"), "/page/gw")
	record2, _ := records.MakeRecord(hfEmbedder, []byte("all work and no play makes jack a dull boy all work and no play makes jack a dull boy all work and..."), "/page/shining")
	record3, _ := records.MakeRecord(hfEmbedder, []byte("What are we having for supper?"), "/page/supper")
	collection := Collection{Id: "test-cosine-similarity", Embedder: hfEmbedder, Records: make(map[string]records.Record)}

	collection.AddRecord(record1)
	collection.AddRecord(record2)
	collection.AddRecord(record3)
	queryResult, err := collection.Query([]byte("Abraham Lincoln, Thomas Jefferson, John F Kennedy"), 1)
	if err != nil {
		t.Errorf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*record1, (*queryResult)[0]) {
		t.Errorf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}

	queryResult, err = collection.Query([]byte("The Shining"), 1)
	if err != nil {
		t.Errorf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*record2, (*queryResult)[0]) {
		t.Errorf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}

	queryResult, err = collection.Query([]byte("We are having chicken and rice for supper, with a side of salad"), 1)
	if err != nil {
		t.Errorf("Query failed: %v", err)
	}
	if !reflect.DeepEqual(*record3, (*queryResult)[0]) {
		t.Errorf("Bad query result. It's possible the embedder sucks, but more likely something is wrong with the library code\n")
	}
}
