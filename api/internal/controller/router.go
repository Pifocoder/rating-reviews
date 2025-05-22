package controller

import (
	"log"
	"net/http"
)

type Server struct {
	addr string
	mux  *http.ServeMux
}

func NewServer(addr string) *Server {
	return &Server{
		addr: addr,
		mux:  http.NewServeMux(),
	}
}

func (s *Server) RegisterHandler(pattern string, handler http.Handler) {
	s.mux.Handle(pattern, handler)
}

func (s *Server) Start() error {
	log.Printf("Starting server on %s\n", s.addr)
	return http.ListenAndServe(s.addr, s.mux)
}
