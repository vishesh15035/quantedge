# QuantEdge 📈

> A high-performance quantitative trading and backtesting platform combining C++ execution engines, Python ML strategies, MLOps pipelines, and a modern frontend — all containerized with Docker.

---

##  Overview

**QuantEdge** is an end-to-end algorithmic trading infrastructure built for speed, modularity, and production readiness. It bridges the gap between research and live trading by integrating:

-  **C++ core engine** for ultra-low-latency order processing and market simulation
-  **Python strategy layer** for signal generation, alpha research, and ML model integration
-  **MLOps pipeline** for model training, versioning, and deployment
-  **Frontend dashboard** for real-time monitoring and strategy visualization
-  **Docker-based deployment** for reproducible, scalable environments

---

##  Architecture

```
quantedge/
├── cpp/              # High-performance C++ execution engine & market simulator
├── python/           # Trading strategies, signal generation, data pipelines
├── mlops/            # ML model training, tracking, and deployment workflows
├── frontend/         # Web dashboard (JavaScript/HTML)
├── docker/           # Docker configuration files
├── .github/
│   └── workflows/    # CI/CD pipelines
├── docker-compose.yml
├── requirements.txt
└── run_backtest.py   # Backtesting entry point
```

---

##  Features

- **Backtesting Engine** — Run historical simulations with tick-level precision via `run_backtest.py`
- **C++ Execution Core** — High-throughput order routing and market microstructure modeling
- **ML Strategy Framework** — Train, evaluate, and deploy ML-based trading strategies
- **MLOps Integration** — Automated model lifecycle management with experiment tracking
- **Live Dashboard** — Interactive frontend to monitor PnL, positions, and strategy metrics
- **CI/CD Ready** — GitHub Actions workflows for automated testing and deployment
- **Fully Dockerized** — One-command setup with `docker-compose`

---

##  Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- Python 3.9+
- A C++ compiler (GCC 11+ or Clang 13+)

### 1. Clone the Repository

```bash
git clone https://github.com/vishesh15035/quantedge.git
cd quantedge
```

### 2. Start All Services with Docker Compose

```bash
docker-compose up --build
```

This spins up the backend engine, MLOps services, and the frontend dashboard together.

### 3. Install Python Dependencies (for local development)

```bash
pip install -r requirements.txt
```

### 4. Run a Backtest

```bash
python run_backtest.py
```

---

##  Running a Strategy

```python
# Example: Run a custom strategy backtest
python run_backtest.py \
  --strategy moving_average_crossover \
  --symbol AAPL \
  --start 2022-01-01 \
  --end 2023-12-31
```

---

## 🤖 MLOps Pipeline

The `mlops/` directory contains tools for:

- **Model Training** — Train predictive models on historical market data
- **Experiment Tracking** — Log metrics, parameters, and artifacts
- **Model Registry** — Version and stage models for deployment
- **Inference Serving** — Deploy models as low-latency prediction services

---

##  C++ Engine

The `cpp/` module implements:

- Order book simulation
- Market microstructure modeling
- Low-latency execution logic
- Real-time risk checks

To build the C++ components manually:

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

##  Frontend Dashboard

The frontend provides:

- Real-time PnL and drawdown charts
- Active positions and order book view
- Strategy performance metrics
- Model monitoring

Access at `http://localhost:3000` after running `docker-compose up`.

---

##  Testing

```bash
# Python tests
pytest python/tests/

# C++ tests (after building)
cd cpp/build && ctest
```

---

##  CI/CD

GitHub Actions workflows in `.github/workflows/` handle:

- Automated Python and C++ testing on push
- Docker image builds and publishing
- Linting and code quality checks

---

##  Roadmap

- [ ] Live broker integration (Alpaca, Interactive Brokers)
- [ ] Real-time market data ingestion (WebSocket feeds)
- [ ] Portfolio-level risk management
- [ ] Strategy parameter optimization (grid search / Bayesian)
- [ ] Enhanced frontend analytics

---

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vishesh** — [@vishesh15035](https://github.com/vishesh15035)

---
