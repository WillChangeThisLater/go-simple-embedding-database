package database

import (
	"errors"
	"fmt"
	"sync"

	collection "go-simple-embedding-database/collection"
	records "go-simple-embedding-database/records"
)

type DataBase interface {
	AddCollection(collection *collection.Collection) error
	DeleteCollection(collectionId string) error
	GetCollection(collectionId string) (*collection.Collection, error)

	AddRecord(collectionId string, record *records.Record) error
	GetRecord(collectionId string, recordId string) error
	DeleteRecord(collectionId string, recordId string) error

	Query(collectionId string, query []byte, n_greatest int) []*records.Record
}

// TODO: if there's a way to refactor this, do it. It's incredibly ugly.
// Specifically, I don't have a good way to pluck the max N elements
// from a list
func (db SimpleDataBase) Query(collectionId string, query []byte, n_greatest int) (*[]records.Record, error) {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return nil, err
	}
	return collection.Query(query, n_greatest)
}

type SimpleDataBase struct {
	mutex       *sync.Mutex                      // TODO: leverage this
	Collections map[string]collection.Collection `json:"collections"`
}

func (db SimpleDataBase) AddRecord(collectionId string, record *records.Record) error {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return err
	}
	return collection.AddRecord(record)
}

func (db SimpleDataBase) GetRecord(collectionId string, recordId string) (*records.Record, error) {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return nil, err
	}
	return collection.GetRecord(recordId)
}

func (db SimpleDataBase) DeleteRecord(collectionId string, recordId string) error {
	collection, err := db.GetCollection(collectionId)
	if err != nil {
		return err
	}
	return collection.DeleteRecord(recordId)
}

func (db SimpleDataBase) AddCollection(collection *collection.Collection) error {
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

func (db SimpleDataBase) isCollectionInDB(collectionId string) bool {
	collections := db.Collections
	_, ok := collections[collectionId]
	return ok
}

func (db SimpleDataBase) GetCollection(collectionId string) (*collection.Collection, error) {
	collection, ok := db.Collections[collectionId]
	if ok {
		return &collection, nil
	}
	return nil, errors.New(fmt.Sprintf("Could not get collection - no collection with ID %s exists in the database", collectionId))
}

func (db SimpleDataBase) DeleteCollection(collectionId string) error {
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

func (db SimpleDataBase) GetCollections() map[string]collection.Collection {
	// I think the locking here is needed?
	db.mutex.Lock()
	defer db.mutex.Unlock()
	return db.Collections
}
