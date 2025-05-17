package api

import "rating-reviews/internal/model/entity"

type CommentRequest struct {
	Text string `json:"text"`
}

type CategoriesResponse struct {
	entity.Categories
}
