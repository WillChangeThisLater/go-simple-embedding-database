package database

import (
	"sync"
	"testing"

	collection "go-simple-embedding-database/collection"
)

func TestDatabaseCollectionAPI(t *testing.T) {
	db := SimpleDataBase{mutex: &sync.Mutex{}, Collections: make(map[string]collection.Collection)}
	if len(db.Collections) > 0 {
		t.Errorf("Database should have no records yet")
	}
	collection1 := collection.MakeCollection("test1", nil)
	err := db.AddCollection(collection1)
	if err != nil {
		t.Errorf("Error adding collection1: %v", err)
	}
	err = db.AddCollection(collection1)
	if err == nil {
		t.Errorf("Should not have been able to add duplicate collection")
	}

	collection2 := collection.MakeCollection("test2", nil)
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
