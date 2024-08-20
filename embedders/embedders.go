package embedders

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
)

type Embedder interface {
	Embed(blob []byte) ([]float64, error)
}

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

type HuggingFaceRequestOptions struct {
	UseCache     bool `json:"use_cache"`
	WaitForModel bool `json:"wait_for_model"`
}

type HuggingFaceRequestBody struct {
	Inputs []string                  `json:"inputs"`
	Value  HuggingFaceRequestOptions `json:"options"`
}

func (e HuggingFaceEmbedder) Embed(blob []byte) ([]float64, error) {
	modelId := e.ModelId

	apiKey := os.Getenv("HUGGING_FACE_API_KEY")
	if apiKey == "" {
		return nil, errors.New("HUGGING_FACE_API_KEY environment variable not set.")
	}
	endpoint := "https://api-inference.huggingface.co/pipeline/feature-extraction"

	body := HuggingFaceRequestBody{Inputs: []string{string(blob)}, Value: HuggingFaceRequestOptions{UseCache: true, WaitForModel: true}}
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
