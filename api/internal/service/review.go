package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go.uber.org/zap"
	"io"
	"math"
	"net/http"
	"rating-reviews/internal/model/api"
	"rating-reviews/internal/model/entity"
	"strings"
)

type Config struct {
	BaseURL string
}

type Analyzer struct {
	logger *zap.SugaredLogger
	config Config
}

func NewAnalyzer(logger *zap.SugaredLogger, cfg Config) Reviewer {
	return &Analyzer{
		logger: logger,
		config: cfg,
	}
}

func (r *Analyzer) Analyze(ctx context.Context, texts []string) (*entity.Keywords, error) {
	combinedText := strings.Join(texts, " ")
	keywords, err := r.getKeywords(ctx, combinedText)
	if err != nil {
		r.logger.Errorf("Ошибка при получении ключевых слов: %v", err)
		return nil, err
	}

	type scoreData struct {
		score  float64
		number int
	}
	result := make(map[string]scoreData, len(keywords))
	for _, word := range keywords {
		result[word] = scoreData{score: 0, number: 0}
	}

	for _, text := range texts {
		prediction, err := r.getPrediction(ctx, text)
		if err != nil {
			r.logger.Warnf("Ошибка при получении предсказания: %v", err)
			continue
		}

		currentKeywords, err := r.getKeywords(ctx, text)
		if err != nil {
			r.logger.Errorf("Ошибка при получении ключевых слов: %v", err)
			return nil, err
		}

		for _, word := range currentKeywords {
			if data, ok := result[word]; ok {
				result[word] = scoreData{
					score:  data.score + prediction,
					number: data.number + 1,
				}
			}
		}
	}

	resultKeywords := entity.Keywords{
		Keywords: make([]entity.Keyword, 0, len(result)),
	}
	for key, data := range result {
		avgScore := 0.0
		if data.number > 0 {
			avgScore = data.score / float64(data.number)
			avgScore = math.Round(avgScore*10) / 10
		}
		resultKeywords.Keywords = append(resultKeywords.Keywords, entity.Keyword{
			Name:  key,
			Score: avgScore,
		})
	}

	return &resultKeywords, nil
}

func (r *Analyzer) getKeywords(ctx context.Context, text string) ([]string, error) {
	url := fmt.Sprintf("%s/keywords", r.config.BaseURL)
	reqBody := entity.Text{Text: text}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("статус %d: %s", resp.StatusCode, string(body))
	}

	var keywordsResp api.KeywordsModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&keywordsResp); err != nil {
		return nil, err
	}
	return keywordsResp.Keywords, nil
}

func (r *Analyzer) getPrediction(ctx context.Context, text string) (float64, error) {
	url := fmt.Sprintf("%s/predict", r.config.BaseURL)
	reqBody := entity.Text{Text: text}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("статус %d: %s", resp.StatusCode, string(body))
	}

	var predResp api.PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predResp); err != nil {
		return 0, err
	}
	return predResp.Prediction, nil
}
