package service

import (
	"context"
	"rating-reviews/internal/model/entity"
)

type Reviewer interface {
	Analyze(ctx context.Context, text []string) (*entity.Keywords, error)
}
