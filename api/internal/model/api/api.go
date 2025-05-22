package api

import "rating-reviews/internal/model/entity"

type CommentRequest struct {
	Text []string `json:"texts"`
}

type CategoriesResponse struct {
	entity.Keywords
}

type KeywordsModelResponse struct {
	Keywords []string `json:"keywords"`
}
type PredictionResponse struct {
	Prediction float64 `json:"prediction"`
}
