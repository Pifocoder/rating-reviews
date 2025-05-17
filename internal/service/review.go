package service

import (
	"go.uber.org/zap"
	"rating-reviews/internal/model/entity"
)

type Analyzer struct {
	logger *zap.SugaredLogger
}

func (r Analyzer) Analyze(text string) (entity.Categories, error) {
	return entity.Categories{}, nil
}

func NewAnalyzer(logger *zap.SugaredLogger) Reviewer {
	return Analyzer{logger: logger}
}
