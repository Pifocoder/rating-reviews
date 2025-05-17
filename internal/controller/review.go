package controller

import (
	"encoding/json"
	"net/http"
	"rating-reviews/internal/model/api"
	"rating-reviews/internal/model/entity"
	"rating-reviews/internal/service"
)

type AnalyzeHandler struct {
	Reviewer *service.Reviewer
}

func (h *AnalyzeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req api.CommentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request payload", http.StatusBadRequest)
		return
	}

	if len(req.Text) < 5 {
		http.Error(w, "text field is required", http.StatusBadRequest)
		return
	}

	categories, err := (*h.Reviewer).Analyze(req.Text)
	if err != nil {
		http.Error(w, "failed to analyze comment", http.StatusInternalServerError)
		return
	}

	resp := api.CategoriesResponse{
		Categories: entity.Categories{
			Categories: make([]entity.Category, len(categories.Categories)),
		},
	}
	for i, c := range categories.Categories {
		resp.Categories.Categories[i] = c
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
		return
	}
}
