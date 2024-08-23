package database

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sync"
	"testing"

	collection "go-simple-embedding-database/collection"
	embedders "go-simple-embedding-database/embedders"
	records "go-simple-embedding-database/records"
)

func MockEmbed(blob []byte) ([]float64, error) {
	return []float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil
}

func TestJSONIO(t *testing.T) {
	tempFileName := "/tmp/testJSONIO.json"
	embedders.EmbedderRegister["fake-embedder"] = MockEmbed
	db := MakeDatabase()
	coll, err := collection.MakeCollection("collection-1", "fake-embedder")
	if err != nil {
		t.Errorf("Could not create collection: %v", err)
	}
	record, err := records.MakeRecord("fake-embedder", []byte("hey there"), "record-1")
	if err != nil {
		t.Errorf("Could not create record: %v", err)
	}
	err = db.AddCollection(coll)
	if err != nil {
		t.Errorf("Could not add collection to database: %v", err)
	}
	err = db.AddRecord("collection-1", record)
	if err != nil {
		t.Errorf("Could not add record to collection: %v", err)
	}
	err = db.ToFile(tempFileName)
	if err != nil {
		t.Errorf("Could not write database to file: %v", err)
	}
	newDB := &SimpleDataBase{}
	err = newDB.FromFile(tempFileName)
	if err != nil {
		t.Errorf("Could not read database from file: %v", err)
	}
	if !reflect.DeepEqual(db, newDB) {
		t.Errorf("Not equal (expected %v, got %v)\n", db, newDB)
	}
}

func TestJSON(t *testing.T) {
	db := MakeDatabase()

	embedders.EmbedderRegister["mock-embed"] = MockEmbed
	collection, err := collection.MakeCollection("test-collection-id", "mock-embed")
	if err != nil {
		t.Errorf("Could not create test collection: %v", err)
	}
	err = db.AddCollection(collection)
	if err != nil {
		t.Errorf("Could not add test collection to database: %v", err)
	}

	record, err := records.MakeRecord("mock-embed", []byte("blob"), "test-record-id")
	if err != nil {
		t.Errorf("Could not create test record: %v", err)
	}
	err = db.AddRecord("test-collection-id", record)
	if err != nil {
		t.Errorf("Could not add record to database: %v", err)
	}

	JSONBody, err := json.Marshal(db)
	if err != nil {
		t.Errorf("Could not marshal JSON for database: %v", err)
	}
	ExpectedJSONBody := []byte("{\"collections\":{\"test-collection-id\":{\"id\":\"test-collection-id\",\"embedderId\":\"mock-embed\",\"embeddings\":{\"test-record-id\":{\"blob\":\"blob\",\"embedding\":[1,2,3,4,5],\"embedderId\":\"mock-embed\",\"id\":\"test-record-id\"}}}}}")
	if !bytes.Equal(JSONBody, ExpectedJSONBody) {
		t.Errorf("Failure marshaling JSON: expected %v, got %v", string(ExpectedJSONBody), string(JSONBody))
	}

	newDB := &SimpleDataBase{}
	err = json.Unmarshal(JSONBody, newDB)
	if err != nil {
		t.Errorf("Could not unmarshal JSON: %v", err)
	}
	if !reflect.DeepEqual(newDB, db) {
		t.Errorf("Unmarshaled DB not equal (expected %v, got %v)", db, newDB)
	}
}

func TestDatabaseCollectionAPI(t *testing.T) {
	db := SimpleDataBase{mutex: &sync.Mutex{}, Collections: make(map[string]collection.Collection)}
	if len(db.Collections) > 0 {
		t.Errorf("Database should have no records yet")
	}

	embedders.EmbedderRegister["mock-embedder"] = MockEmbed
	collection1, err := collection.MakeCollection("test1", "mock-embedder")
	if err != nil {
		t.Errorf("Could not create collection1: %v", err)
	}
	err = db.AddCollection(collection1)
	if err != nil {
		t.Errorf("Error adding collection1: %v", err)
	}
	err = db.AddCollection(collection1)
	if err == nil {
		t.Errorf("Should not have been able to add duplicate collection")
	}

	collection2, err := collection.MakeCollection("test2", "mock-embedder")
	if err != nil {
		t.Errorf("Could not create collection2: %v", err)
	}

	err = db.AddCollection(collection2)
	if err != nil {
		t.Errorf("Error adding collection2: %v", err)
	}
	if len(db.Collections) != 2 {
		t.Errorf("Should have 2 collections; have %d", len(db.Collections))
	}

	err = db.DeleteCollection(collection1.Id)
	if err != nil {
		t.Errorf("Error adding collection1: %v", err)
	}
	err = db.DeleteCollection(collection2.Id)
	if err != nil {
		t.Errorf("Error adding collection2: %v", err)
	}

	err = db.DeleteCollection(collection1.Id)
	if err == nil {
		t.Errorf("Should not be able to delete collection1 as it was already deleted")
	}
}
