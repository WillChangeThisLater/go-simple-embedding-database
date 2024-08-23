A package implementing a very simple, Go-based vector embedding database.

The interface is very similar to ChromaDB.

Here's an example:

```go
package main

import (
	"fmt"
	"io"
	"net/http"

	collection "go-simple-embedding-database/collection"
	database "go-simple-embedding-database/database"
	records "go-simple-embedding-database/records"
)

func main() {

	// read in romeo and juliet
	rj := "https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/romeo-and-juliet_TXT_FolgerShakespeare.txt"
	text, err := http.Get(rj)
	if err != nil {
		panic(err)
	}
	defer text.Body.Close()
	buffer, err := io.ReadAll(text.Body)
	if err != nil {
		panic(err)
	}

	// setup the database
        //
        // you'll need HUGGING_FACE_API_KEY set up in your environment for this to work
        // it takes about a minute to run on my (not beefy) machine
	embedderId := "hugging-face/sentence-transformers/all-MiniLM-L12-v1"
	collectionId := "romeo-and-juliet"
	db := database.MakeDatabase()
	collection, err := collection.MakeCollection(collectionId, embedderId)
	if err != nil {
		panic(err)
	}
	err = db.AddCollection(collection)
	if err != nil {
		panic(err)
	}

	chars := len(buffer)
	chunkSize := 4096
	chunkNum := 0
	for i := 0; i < chars; i += chunkSize {
		fmt.Printf("Processing chunk %d of %d\n", chunkNum+1, chars/chunkSize)
		chunk := buffer[i:max(chars, i+chunkSize)]
		record, err := records.MakeRecord(embedderId, chunk, fmt.Sprintf("rj-%d", chunkNum))
		if err != nil {
			panic(err)
		}
		err = db.AddRecord(collectionId, record)
		if err != nil {
			panic(err)
		}
		chunkNum += 1
	}

	scenes, err := db.Query(collectionId, []byte("that famous scene where juliet asks romeo where he is"), 1)
	famousScene := (*scenes)[0]
	if err != nil {
		panic(err)
	}
	fmt.Printf("Here's the famous scene: %s\n", string(famousScene.Blob))

    // if you like, you can write out the results to a JSON file
    //
    // db.ToFile("rj.json")
    //
    // or read them in from a file
    //
    // newDB := &database.SimpleDataBase{}
    // newDB.FromFile("rj.json")
}
```

Adding an embedding is really easy. All you need is a function matching the 
following signature

```go
func (blob []byte) ([]float64, error)
```

You can register the function with the embedders module using a embedding id.
For example,

```go
package main

import (
	"fmt"

	embedders "go-simple-embedding-database/embedders"
	records "go-simple-embedding-database/records"
)

func mockEmbed(blob []byte) ([]float64, error) {
	return []float64{3.14159}, nil
}

func main() {
	embedders.EmbedderRegister["mock-embedder"] = mockEmbed
	recordId := "record-0"
	record, err := records.MakeRecord("mock-embedder", []byte("blah blah blah..."), recordId)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Embedding for record %s is %v\n", recordId, record.Embedding)
}
```
