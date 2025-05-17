package service

import "rating-reviews/internal/model/entity"

type Reviewer interface {
	Analyze(text string) (entity.Categories, error)
}
