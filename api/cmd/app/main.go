package main

import (
	"flag"
	"log"
	"rating-reviews/cmd"
	"rating-reviews/internal/config"
)

func main() {
	var cfgPath string
	flag.StringVar(&cfgPath, "config", "/etc/rating-reviews/config.yaml", "path to config file")
	flag.Parse()

	cfg := config.NewConfig()
	err := cfg.Load(cfgPath)
	if err != nil {
		log.Fatalf("Error load config: %v", err)
		return
	}

	a := cmd.NewApp()
	a.Run(*cfg)
}
