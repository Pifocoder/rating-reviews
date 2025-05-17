package cmd

import (
	"go.uber.org/zap"
	"log"
	"os"
	"os/signal"
	"rating-reviews/internal/config"
	"rating-reviews/internal/controller"
	"rating-reviews/internal/service"
	"syscall"
)

type App struct {
}

func NewApp() *App {
	return &App{}
}

func (*App) Run(cfg config.Config) {
	server := controller.NewServer(cfg.Addr)
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatal("problem with logger")
		return
	}

	analyzer := service.NewAnalyzer(logger)
	handler := &controller.AnalyzeHandler{Reviewer: &analyzer}
	server.RegisterHandler("/analyze", handler)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		server.Start()
	}()

	<-stop
}
