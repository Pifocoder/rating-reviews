# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей и устанавливаем их
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY . .

# Открываем порт 80 для сервера
EXPOSE 80

# Запускаем Uvicorn с привязкой к 0.0.0.0 и порту 80
CMD /bin/sh -c "cat x* > best_model_val_acc_phase2_CrossEntropyLoss.pt && uvicorn main:app --host 0.0.0.0 --port 80"
