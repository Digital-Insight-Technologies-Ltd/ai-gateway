// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/api"
)

// GCP Vertex AI Embeddings API request/response types.
// https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

type gcpEmbeddingRequest struct {
	Instances  []gcpEmbeddingInstance  `json:"instances"`
	Parameters *gcpEmbeddingParameters `json:"parameters,omitempty"`
}

type gcpEmbeddingInstance struct {
	Content string `json:"content"`
}

type gcpEmbeddingParameters struct {
	OutputDimensionality *int `json:"outputDimensionality,omitempty"`
}

type gcpEmbeddingResponse struct {
	Predictions []gcpEmbeddingPrediction `json:"predictions"`
	Metadata    *gcpEmbeddingMetadata    `json:"metadata,omitempty"`
}

type gcpEmbeddingPrediction struct {
	Embeddings gcpEmbeddingValues `json:"embeddings"`
}

type gcpEmbeddingValues struct {
	Values     []float64          `json:"values"`
	Statistics *gcpEmbeddingStats `json:"statistics,omitempty"`
}

type gcpEmbeddingStats struct {
	TokenCount int  `json:"token_count"`
	Truncated  bool `json:"truncated"`
}

type gcpEmbeddingMetadata struct {
	BillableCharacterCount int `json:"billableCharacterCount,omitempty"`
}

// NewEmbeddingOpenAIToGCPVertexAITranslator implements [Factory] for OpenAI to GCP Vertex AI translation for embeddings.
func NewEmbeddingOpenAIToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) OpenAIEmbeddingTranslator {
	return &openAIToGCPVertexAITranslatorV1Embedding{modelNameOverride: modelNameOverride}
}

// openAIToGCPVertexAITranslatorV1Embedding translates OpenAI Embeddings API to GCP Vertex AI Embeddings API.
// https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
type openAIToGCPVertexAITranslatorV1Embedding struct {
	modelNameOverride internalapi.ModelNameOverride
	requestModel      internalapi.RequestModel
}

// RequestBody implements [OpenAIEmbeddingTranslator.RequestBody] for GCP Vertex AI.
func (o *openAIToGCPVertexAITranslatorV1Embedding) RequestBody(_ []byte, req *openai.EmbeddingRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	o.requestModel = req.Model
	if o.modelNameOverride != "" {
		o.requestModel = o.modelNameOverride
	}

	instances, err := o.convertInput(req.Input)
	if err != nil {
		return nil, nil, err
	}

	gcpReq := gcpEmbeddingRequest{Instances: instances}
	if req.Dimensions != nil {
		gcpReq.Parameters = &gcpEmbeddingParameters{
			OutputDimensionality: req.Dimensions,
		}
	}

	path := buildGCPModelPathSuffix(gcpModelPublisherGoogle, o.requestModel, gcpMethodPredict)
	newBody, err = json.Marshal(gcpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal GCP embedding request: %w", err)
	}

	newHeaders = []internalapi.Header{
		{pathHeaderName, path},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// convertInput converts OpenAI embedding input to GCP embedding instances.
func (o *openAIToGCPVertexAITranslatorV1Embedding) convertInput(input openai.EmbeddingRequestInput) ([]gcpEmbeddingInstance, error) {
	var instances []gcpEmbeddingInstance

	switch v := input.Value.(type) {
	case string:
		if v == "" {
			return nil, fmt.Errorf("empty input is not allowed")
		}
		instances = append(instances, gcpEmbeddingInstance{Content: v})
	case []string:
		for _, s := range v {
			instances = append(instances, gcpEmbeddingInstance{Content: s})
		}
	case []any:
		for _, item := range v {
			switch s := item.(type) {
			case string:
				instances = append(instances, gcpEmbeddingInstance{Content: s})
			default:
				return nil, fmt.Errorf("token array input is not supported for GCP Vertex AI embeddings")
			}
		}
	case []int64, [][]int64:
		return nil, fmt.Errorf("token array input is not supported for GCP Vertex AI embeddings")
	default:
		return nil, fmt.Errorf("unsupported input type for GCP Vertex AI embeddings: %T", v)
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("empty input is not allowed")
	}

	return instances, nil
}

// ResponseHeaders implements [OpenAIEmbeddingTranslator.ResponseHeaders].
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	return nil, nil
}

// ResponseBody implements [OpenAIEmbeddingTranslator.ResponseBody] for GCP Vertex AI.
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracing.EmbeddingsSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	var gcpResp gcpEmbeddingResponse
	if err = json.NewDecoder(body).Decode(&gcpResp); err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("failed to decode GCP embedding response: %w", err)
	}

	openAIResp := o.convertResponse(gcpResp)
	if span != nil {
		span.RecordResponse(&openAIResp)
	}

	tokenUsage.SetInputTokens(uint32(openAIResp.Usage.PromptTokens)) //nolint:gosec
	tokenUsage.SetTotalTokens(uint32(openAIResp.Usage.TotalTokens))  //nolint:gosec

	newBody, err = json.Marshal(openAIResp)
	if err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("failed to marshal OpenAI embedding response: %w", err)
	}

	newHeaders = []internalapi.Header{
		{contentTypeHeaderName, jsonContentType},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}

	responseModel = o.requestModel
	return
}

func (o *openAIToGCPVertexAITranslatorV1Embedding) convertResponse(gcpResp gcpEmbeddingResponse) openai.EmbeddingResponse {
	embeddings := make([]openai.Embedding, 0, len(gcpResp.Predictions))
	totalTokens := 0

	for i, prediction := range gcpResp.Predictions {
		embeddings = append(embeddings, openai.Embedding{
			Object:    "embedding",
			Embedding: openai.EmbeddingUnion{Value: prediction.Embeddings.Values},
			Index:     i,
		})
		if prediction.Embeddings.Statistics != nil {
			totalTokens += prediction.Embeddings.Statistics.TokenCount
		}
	}

	return openai.EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  o.requestModel,
		Usage: openai.EmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}
}

// ResponseError implements [Translator.ResponseError] for GCP Vertex AI.
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseError(respHeaders map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	statusCode := respHeaders[statusHeaderName]

	if v, ok := respHeaders[contentTypeHeaderName]; ok && strings.Contains(v, jsonContentType) {
		var gcpErr gcpVertexAIError
		if decodeErr := json.NewDecoder(body).Decode(&gcpErr); decodeErr == nil && gcpErr.Error.Message != "" {
			openaiError := openai.Error{
				Type: "error",
				Error: openai.ErrorType{
					Type:    gcpErr.Error.Status,
					Message: gcpErr.Error.Message,
					Code:    &statusCode,
				},
			}
			if openaiError.Error.Type == "" {
				openaiError.Error.Type = gcpVertexAIBackendError
			}
			newBody, err = json.Marshal(openaiError)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to marshal error response: %w", err)
			}
			newHeaders = []internalapi.Header{
				{contentTypeHeaderName, jsonContentType},
				{contentLengthHeaderName, strconv.Itoa(len(newBody))},
			}
			return newHeaders, newBody, nil
		}
	}

	buf, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read error body: %w", err)
	}
	openaiError := openai.Error{
		Type: "error",
		Error: openai.ErrorType{
			Type:    gcpVertexAIBackendError,
			Message: string(buf),
			Code:    &statusCode,
		},
	}
	newBody, err = json.Marshal(openaiError)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal error body: %w", err)
	}
	newHeaders = []internalapi.Header{
		{contentTypeHeaderName, jsonContentType},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}
