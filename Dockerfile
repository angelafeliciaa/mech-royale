FROM python:3.13.18-slim
WORKDIR /bot
COPY *.py ./
ENTRYPOINT ["python3", "-u", "/bot/dex_bot.py"]
