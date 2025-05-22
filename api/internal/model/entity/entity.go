package entity

type Keyword struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

type Keywords struct {
	Keywords []Keyword `json:"keywords"`
}

type Text struct {
	Text string `json:"text"`
}
