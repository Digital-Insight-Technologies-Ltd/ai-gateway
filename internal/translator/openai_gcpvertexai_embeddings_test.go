// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
)

func TestNewEmbeddingOpenAIToGCPVertexAITranslator(t *testing.T) {
	translator := NewEmbeddingOpenAIToGCPVertexAITranslator("")
	require.NotNil(t, translator)

	translatorWithOverride := NewEmbeddingOpenAIToGCPVertexAITranslator("custom-model")
	require.NotNil(t, translatorWithOverride)
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingRequestBody(t *testing.T) {
	for _, tc := range []struct {
		name              string
		originalBody      string
		modelNameOverride internalapi.ModelNameOverride
		expPath           string
		expError          bool
		expErrorContains  string
		validateBody      func(t *testing.T, body []byte)
	}{
		{
			name:         "single_string_input",
			originalBody: `{"model":"text-embedding-004","input":"hello world"}`,
			expPath:      "publishers/google/models/text-embedding-004:predict",
			validateBody: func(t *testing.T, body []byte) {
				var gcpReq gcpEmbeddingRequest
				require.NoError(t, json.Unmarshal(body, &gcpReq))
				require.Len(t, gcpReq.Instances, 1)
				require.Equal(t, "hello world", gcpReq.Instances[0].Content)
				require.Nil(t, gcpReq.Parameters)
			},
		},
		{
			name:         "array_string_input",
			originalBody: `{"model":"text-embedding-004","input":["hello","world"]}`,
			expPath:      "publishers/google/models/text-embedding-004:predict",
			validateBody: func(t *testing.T, body []byte) {
				var gcpReq gcpEmbeddingRequest
				require.NoError(t, json.Unmarshal(body, &gcpReq))
				require.Len(t, gcpReq.Instances, 2)
				require.Equal(t, "hello", gcpReq.Instances[0].Content)
				require.Equal(t, "world", gcpReq.Instances[1].Content)
			},
		},
		{
			name:         "with_dimensions",
			originalBody: `{"model":"text-embedding-004","input":"test","dimensions":256}`,
			expPath:      "publishers/google/models/text-embedding-004:predict",
			validateBody: func(t *testing.T, body []byte) {
				var gcpReq gcpEmbeddingRequest
				require.NoError(t, json.Unmarshal(body, &gcpReq))
				require.NotNil(t, gcpReq.Parameters)
				require.NotNil(t, gcpReq.Parameters.OutputDimensionality)
				require.Equal(t, 256, *gcpReq.Parameters.OutputDimensionality)
			},
		},
		{
			name:              "model_name_override",
			originalBody:      `{"model":"text-embedding-004","input":"test"}`,
			modelNameOverride: "custom-embedding-model",
			expPath:           "publishers/google/models/custom-embedding-model:predict",
		},
		{
			name:             "token_array_input",
			originalBody:     `{"model":"text-embedding-004","input":[1,2,3]}`,
			expError:         true,
			expErrorContains: "token array input is not supported",
		},
		{
			name:             "empty_input",
			originalBody:     `{"model":"text-embedding-004","input":""}`,
			expError:         true,
			expErrorContains: "empty input is not allowed",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator := NewEmbeddingOpenAIToGCPVertexAITranslator(tc.modelNameOverride)
			var req openai.EmbeddingRequest
			require.NoError(t, json.Unmarshal([]byte(tc.originalBody), &req))

			headerMutation, bodyMutation, err := translator.RequestBody([]byte(tc.originalBody), &req, false)

			if tc.expError {
				require.Error(t, err)
				if tc.expErrorContains != "" {
					require.Contains(t, err.Error(), tc.expErrorContains)
				}
				return
			}

			require.NoError(t, err)
			require.NotNil(t, headerMutation)
			require.GreaterOrEqual(t, len(headerMutation), 1)
			require.Equal(t, pathHeaderName, headerMutation[0].Key())
			require.Equal(t, tc.expPath, headerMutation[0].Value())

			require.NotNil(t, bodyMutation)
			require.Len(t, headerMutation, 2)
			require.Equal(t, contentLengthHeaderName, headerMutation[1].Key())

			if tc.validateBody != nil {
				tc.validateBody(t, bodyMutation)
			}
		})
	}
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingRequestBodyOnRetry(t *testing.T) {
	translator := NewEmbeddingOpenAIToGCPVertexAITranslator("")
	originalBody := `{"model":"text-embedding-004","input":"test"}`
	var req openai.EmbeddingRequest
	require.NoError(t, json.Unmarshal([]byte(originalBody), &req))

	headerMutation, bodyMutation, err := translator.RequestBody([]byte(originalBody), &req, true)
	require.NoError(t, err)
	require.NotNil(t, headerMutation)
	require.NotNil(t, bodyMutation)
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingResponseHeaders(t *testing.T) {
	translator := NewEmbeddingOpenAIToGCPVertexAITranslator("")
	headerMutation, err := translator.ResponseHeaders(map[string]string{})
	require.NoError(t, err)
	require.Nil(t, headerMutation)
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingResponseBody(t *testing.T) {
	for _, tc := range []struct {
		name             string
		responseBody     string
		expTokenUsage    metrics.TokenUsage
		expError         bool
		expModel         string
		validateResponse func(t *testing.T, body []byte)
	}{
		{
			name: "single_embedding",
			responseBody: `{
				"predictions": [
					{
						"embeddings": {
							"values": [0.1, 0.2, 0.3],
							"statistics": {"token_count": 5, "truncated": false}
						}
					}
				]
			}`,
			expTokenUsage: gcpEmbeddingTokenUsageFrom(5, 5),
			expModel:      "text-embedding-004",
			validateResponse: func(t *testing.T, body []byte) {
				var resp openai.EmbeddingResponse
				require.NoError(t, json.Unmarshal(body, &resp))
				require.Equal(t, "list", resp.Object)
				require.Len(t, resp.Data, 1)
				require.Equal(t, "embedding", resp.Data[0].Object)
				require.Equal(t, 0, resp.Data[0].Index)
				// After JSON round-trip, []float64 becomes []any.
				values, ok := resp.Data[0].Embedding.Value.([]any)
				if !ok {
					// Direct []float64 if not round-tripped.
					valuesFloat, ok := resp.Data[0].Embedding.Value.([]float64)
					require.True(t, ok)
					require.Len(t, valuesFloat, 3)
				} else {
					require.Len(t, values, 3)
				}
			},
		},
		{
			name: "batch_embeddings",
			responseBody: `{
				"predictions": [
					{
						"embeddings": {
							"values": [0.1, 0.2],
							"statistics": {"token_count": 3, "truncated": false}
						}
					},
					{
						"embeddings": {
							"values": [0.3, 0.4],
							"statistics": {"token_count": 4, "truncated": false}
						}
					}
				]
			}`,
			expTokenUsage: gcpEmbeddingTokenUsageFrom(7, 7),
			expModel:      "text-embedding-004",
			validateResponse: func(t *testing.T, body []byte) {
				var resp openai.EmbeddingResponse
				require.NoError(t, json.Unmarshal(body, &resp))
				require.Len(t, resp.Data, 2)
				require.Equal(t, 0, resp.Data[0].Index)
				require.Equal(t, 1, resp.Data[1].Index)
				require.Equal(t, 7, resp.Usage.PromptTokens)
				require.Equal(t, 7, resp.Usage.TotalTokens)
			},
		},
		{
			name: "no_statistics",
			responseBody: `{
				"predictions": [
					{
						"embeddings": {
							"values": [0.1, 0.2, 0.3]
						}
					}
				]
			}`,
			expTokenUsage: gcpEmbeddingTokenUsageFrom(0, 0),
			expModel:      "text-embedding-004",
		},
		{
			name:          "invalid_json",
			responseBody:  `invalid json`,
			expError:      true,
			expTokenUsage: metrics.TokenUsage{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator := NewEmbeddingOpenAIToGCPVertexAITranslator("")
			// Pre-populate the request model.
			originalBody := `{"model":"text-embedding-004","input":"test"}`
			var req openai.EmbeddingRequest
			require.NoError(t, json.Unmarshal([]byte(originalBody), &req))
			_, _, _ = translator.RequestBody([]byte(originalBody), &req, false)

			respHeaders := map[string]string{
				contentTypeHeaderName: jsonContentType,
				statusHeaderName:      "200",
			}

			headerMutation, bodyMutation, tokenUsage, responseModel, err := translator.ResponseBody(
				respHeaders,
				strings.NewReader(tc.responseBody),
				true,
				nil,
			)

			if tc.expError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.expTokenUsage, tokenUsage)

			if tc.expModel != "" {
				require.Equal(t, tc.expModel, responseModel)
			}

			// GCP to OpenAI translation always returns body mutation.
			require.NotNil(t, headerMutation)
			require.NotNil(t, bodyMutation)

			if tc.validateResponse != nil {
				tc.validateResponse(t, bodyMutation)
			}
		})
	}
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingResponseError(t *testing.T) {
	translator := NewEmbeddingOpenAIToGCPVertexAITranslator("")

	t.Run("gcp_json_error", func(t *testing.T) {
		respHeaders := map[string]string{
			statusHeaderName:      "400",
			contentTypeHeaderName: jsonContentType,
		}
		errorBody := `{"error":{"code":400,"message":"Invalid request","status":"INVALID_ARGUMENT"}}`

		headerMutation, bodyMutation, err := translator.ResponseError(respHeaders, strings.NewReader(errorBody))
		require.NoError(t, err)
		require.NotNil(t, headerMutation)
		require.NotNil(t, bodyMutation)

		var openaiError openai.Error
		require.NoError(t, json.Unmarshal(bodyMutation, &openaiError))
		require.Equal(t, "error", openaiError.Type)
		require.Equal(t, "INVALID_ARGUMENT", openaiError.Error.Type)
		require.Equal(t, "Invalid request", openaiError.Error.Message)
		require.Equal(t, "400", *openaiError.Error.Code)
	})

	t.Run("non_json_error", func(t *testing.T) {
		respHeaders := map[string]string{
			statusHeaderName:      "503",
			contentTypeHeaderName: "text/plain",
		}
		errorBody := "Service Unavailable"

		headerMutation, bodyMutation, err := translator.ResponseError(respHeaders, strings.NewReader(errorBody))
		require.NoError(t, err)
		require.NotNil(t, headerMutation)
		require.NotNil(t, bodyMutation)

		var openaiError openai.Error
		require.NoError(t, json.Unmarshal(bodyMutation, &openaiError))
		require.Equal(t, "error", openaiError.Type)
		require.Equal(t, gcpVertexAIBackendError, openaiError.Error.Type)
		require.Equal(t, errorBody, openaiError.Error.Message)
		require.Equal(t, "503", *openaiError.Error.Code)
	})

	t.Run("gcp_error_without_status", func(t *testing.T) {
		respHeaders := map[string]string{
			statusHeaderName:      "500",
			contentTypeHeaderName: jsonContentType,
		}
		errorBody := `{"error":{"code":500,"message":"Internal error"}}`

		headerMutation, bodyMutation, err := translator.ResponseError(respHeaders, strings.NewReader(errorBody))
		require.NoError(t, err)
		require.NotNil(t, headerMutation)
		require.NotNil(t, bodyMutation)

		var openaiError openai.Error
		require.NoError(t, json.Unmarshal(bodyMutation, &openaiError))
		require.Equal(t, gcpVertexAIBackendError, openaiError.Error.Type)
	})
}

func TestOpenAIToGCPVertexAITranslatorV1EmbeddingConvertInput(t *testing.T) {
	translator := &openAIToGCPVertexAITranslatorV1Embedding{}

	t.Run("string_input", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: "hello"}
		instances, err := translator.convertInput(input)
		require.NoError(t, err)
		require.Len(t, instances, 1)
		require.Equal(t, "hello", instances[0].Content)
	})

	t.Run("string_array_input", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: []string{"hello", "world"}}
		instances, err := translator.convertInput(input)
		require.NoError(t, err)
		require.Len(t, instances, 2)
		require.Equal(t, "hello", instances[0].Content)
		require.Equal(t, "world", instances[1].Content)
	})

	t.Run("any_array_with_strings", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: []any{"hello", "world"}}
		instances, err := translator.convertInput(input)
		require.NoError(t, err)
		require.Len(t, instances, 2)
	})

	t.Run("any_array_with_numbers", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: []any{1.0, 2.0, 3.0}}
		_, err := translator.convertInput(input)
		require.Error(t, err)
		require.Contains(t, err.Error(), "token array input is not supported")
	})

	t.Run("empty_string", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: ""}
		_, err := translator.convertInput(input)
		require.Error(t, err)
		require.Contains(t, err.Error(), "empty input is not allowed")
	})

	t.Run("empty_array", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: []string{}}
		_, err := translator.convertInput(input)
		require.Error(t, err)
		require.Contains(t, err.Error(), "empty input is not allowed")
	})

	t.Run("unsupported_type", func(t *testing.T) {
		input := openai.EmbeddingRequestInput{Value: 123}
		_, err := translator.convertInput(input)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unsupported input type")
	})
}

// gcpEmbeddingTokenUsageFrom creates a TokenUsage for GCP embedding tests.
// Embeddings only have input tokens, no output tokens.
func gcpEmbeddingTokenUsageFrom(input, total int32) metrics.TokenUsage {
	var usage metrics.TokenUsage
	if input >= 0 {
		usage.SetInputTokens(uint32(input))
	}
	if total >= 0 {
		usage.SetTotalTokens(uint32(total))
	}
	return usage
}
