FROM python:3.13.18-slim
WORKDIR /bot
COPY dex_bot.py dex_nav.py dex_combat.py dex_tactics.py arena_map.py mechfight.py ./
ENTRYPOINT ["python3", "-u", "/bot/dex_bot.py"]
