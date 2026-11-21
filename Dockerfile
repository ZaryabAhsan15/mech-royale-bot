FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY mechfight.py world.py bot.py ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
