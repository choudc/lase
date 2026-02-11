# Operations

## Start (prod-style)

```bash
./deploy.sh
./start.sh
```

Then open `http://localhost:5000`.

To run in background:

```bash
./start.sh --daemon
```

## Stop

```bash
./stop.sh
```

## Dev (hot reload)

```bash
./dev.sh
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:5000`
