# mech-royale

Bot for the Builders Buildathon **Mech Royale** track (16-mech battle royale, placement scoring).

## Layout

| Path | What |
| --- | --- |
| `Dockerfile` | Image the scorer builds (linux/amd64). |
| `example_bot.py` | Bot entrypoint. |
| `mechfight.py` | Protocol wrapper from the kit (stdlib only). |
| `kit/` | Local-only: conformance check, map, protocol docs. Excluded from the image via `.dockerignore`. |

## Test locally

```sh
python3 kit/check/check.py -- python3 -u example_bot.py
docker build --platform linux/amd64 -t my-bot .
python3 kit/check/check.py --map-dir /arena -- docker run --rm -i --platform linux/amd64 --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/kit/arena:/arena:ro" my-bot
```

## Third-party code

- `mechfight.py`, `example_bot.py` (starting point), and everything in `kit/` come from the Builders `mechfight-kit`.

## License

MIT
