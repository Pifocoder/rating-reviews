package entity

type Category struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

type Categories struct {
	Categories []Category `json:"categories"`
}
