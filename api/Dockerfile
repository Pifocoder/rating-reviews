# Используем официальный образ Go для сборки
FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o rating-reviews ./cmd/app

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/rating-reviews .

COPY config.yaml /etc/rating-reviews/config.yaml

EXPOSE 8080

CMD ["./rating-reviews", "-config", "/etc/rating-reviews/config.yaml"]
