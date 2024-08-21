package collection

import (
	"errors"
	"fmt"
	"reflect"
	"slices"

	embedders "go-simple-embedding-database/embedders"
	records "go-simple-embedding-database/records"
	utils "go-simple-embedding-database/utils"
)

type Collection struct {
	Id       string                    `json:"id"`
	Embedder embedders.Embedder        `json:"embedder"`
	Records  map[string]records.Record `json:"embeddings"`
}

func MakeCollection(id string, embedder embedders.Embedder) *Collection {
	collection := Collection{Id: id, Embedder: embedder, Records: make(map[string]records.Record)}
	return &collection
}

// GPT created this one.
//func (c *Collection) UnmarshalJSON(data []byte) error {
//	// Create a helper struct that mirrors Collection, but with Embedder as a json.RawMessage
//	type Alias Collection
//	helper := &struct {
//		*Alias
//		Embedder json.RawMessage `json:"embedder"`
//	}{
//		Alias: (*Alias)(c),
//	}
//
//	// Unmarshal the JSON into the helper struct
//	if err := json.Unmarshal(data, &helper); err != nil {
//		return err
//	}
//
//	var rawEmbedder map[string]interface{}
//	if err := json.Unmarshal(helper.Embedder, &rawEmbedder); err != nil {
//		return err
//	}
//	id, ok := rawEmbedder["id"].(string)
//	// TODO; write a better error message here.
//	if !ok {
//		return errors.New("Embedder interface does not have an 'id' field, so it cannot be unmarshaled")
//	}
//
//	// Dispatch based on the 'id' field
//	switch id {
//	case "HuggingFaceEmbedder":
//		var embedder embedders.HuggingFaceEmbedder
//		if err := json.Unmarshal(helper.Embedder, &embedder); err != nil {
//			return err
//		}
//		c.Embedder = embedder
//	case "MockEmbedder":
//		var embedder embedders.MockEmbedder
//		if err := json.Unmarshal(helper.Embedder, &embedder); err != nil {
//			return err
//		}
//		c.Embedder = embedder
//	default:
//		return errors.New(fmt.Sprintf("unknown embedder id: %s", id))
//	}
//
//	return nil
//}

func (c Collection) String() string {
	return fmt.Sprintf("Collection{collection.Id: %s, embedder: %v}", c.Id, c.Embedder)
}

func (collection Collection) AddRecord(record *records.Record) error {
	_, ok := collection.Records[record.Id]
	if ok {
		return errors.New(fmt.Sprintf("Record %s already exists in collection %s\n", record.Id, collection.Id))
	}
	if !reflect.DeepEqual(collection.Embedder, record.Embedder) {
		return errors.New(fmt.Sprintf("Record embedder %v != collection embedder %v", record.Embedder, collection.Embedder))
	}
	if record.Embedding == nil {
		return errors.New(fmt.Sprintf("Embedding for %v is null", record))
	}
	collection.Records[record.Id] = *record
	return nil
}

func (collection Collection) DeleteRecord(recordId string) error {
	_, ok := collection.Records[recordId]
	if ok {
		delete(collection.Records, recordId)
		return nil
	}
	return errors.New(fmt.Sprintf("Could not delete record %s from collection %s: record not found in collection", recordId, collection.Id))
}

func (collection Collection) GetRecord(recordId string) (*records.Record, error) {
	record, ok := collection.Records[recordId]
	if !ok {
		return nil, errors.New(fmt.Sprintf("Could not get record - record with ID %s does not exist in collection", recordId))
	}
	return &record, nil
}

func (collection Collection) Query(query []byte, n_greatest int) (*[]records.Record, error) {

	embedder := collection.Embedder
	queryEmbedding, err := embedder.Embed(query)
	if err != nil {
		return nil, err
	}

	// TODO: maybe log out an error here. I can't think of many cases where this is what we'd want to do
	if len(collection.Records) <= n_greatest {
		records := make([]records.Record, 0)
		for _, record := range collection.Records {
			records = append(records, record)
		}
		return &records, nil
	}

	mostSimilarRecords := make([]records.Record, 0)
	similarities := make(map[string]float64)

	// determine how close each record's embedding is to the query's embedding
	for recordId, record := range collection.Records {
		similarity, err := utils.CosineSimilarity(queryEmbedding, record.Embedding)
		if err != nil {
			return nil, err
		}
		similarities[recordId] = similarity
	}

	// basically just the records from 'similarities' map,
	// but sorted in descending order
	distances := make([]float64, 0)
	for _, distance := range similarities {
		distances = append(distances, distance)
	}
	slices.Sort(distances)
	slices.Reverse(distances)

	// this is an ugly hack to deal with potential duplicate values
	finalValue := distances[n_greatest-1]
	if finalValue == distances[n_greatest] {
		numPicked := 0
		for recordId, distance := range similarities {
			if distance > finalValue {
				record, err := collection.GetRecord(recordId)
				if err != nil {
					return nil, err
				}
				mostSimilarRecords = append(mostSimilarRecords, *record)
				numPicked += 1
			}
		}
		for recordId, distance := range similarities {
			if distance == finalValue {
				record, err := collection.GetRecord(recordId)
				if err != nil {
					return nil, err
				}
				mostSimilarRecords = append(mostSimilarRecords, *record)
				numPicked += 1
				if numPicked == n_greatest {
					if len(mostSimilarRecords) != n_greatest {
						return nil, errors.New(fmt.Sprintf("matching - len(mostSimilarRecords) != n_greatest (%d != %d)", len(mostSimilarRecords), n_greatest))
					}
					return &mostSimilarRecords, nil
				}
			}
		}
	}

	// even the straightforward case is a little ugly
	for recordId, distance := range similarities {
		if distance >= finalValue {
			record, err := collection.GetRecord(recordId)
			if err != nil {
				return nil, err
			}
			mostSimilarRecords = append(mostSimilarRecords, *record)
		}
	}
	if len(mostSimilarRecords) != n_greatest {
		return nil, errors.New(fmt.Sprintf("distinct - len(mostSimilarRecords) != n_greatest (%d != %d)", len(mostSimilarRecords), n_greatest))
	}
	return &mostSimilarRecords, nil
}
