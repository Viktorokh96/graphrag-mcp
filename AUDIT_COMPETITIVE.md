# Аудит graphrag (v0.2.0) и конкурентный анализ

## 1. Обзор проекта graphrag

**GraphRAG** — production-grade RAG MCP-сервер с гибридным поиском (dense + sparse + граф), HTTP API и CLI. Работает как MCP-сервер (JSON-RPC через stdio/SSE) или FastAPI REST.

### Технический стек

| Компонент | Технология |
|-----------|-----------|
| Эмбеддинги | BGE-M3 (1024d) / Ollama (4096d) / OpenRouter (1536d) |
| Векторное хранилище | Qdrant (dense + sparse BM25 dual-векторы) |
| Граф | SQLite + NetworkX (in-memory BFS) |
| Документы | SQLite (WAL) / PostgreSQL |
| Ранжирование | CrossEncoder BAAI/bge-reranker-v2-m3 |
| Query expansion | Qwen3-1.8B + RRF |
| Graph extraction | LLM Qwen3-4B + spaCy NER |
| Транспорты | MCP stdio, MCP SSE, HTTP REST, CLI |
| Визуализация | vis.js |
| CI/CD | pytest, ruff |

### Ключевые фичи

- Три режима поиска: semantic (dense), BM25 (sparse), hybrid (RRF alpha-dilution)
- Language-aware alpha (0.85 для кириллицы, 0.5 для остальных)
- CrossEncoder reranker (lazy load)
- Multi-query expansion через Ollama LLM
- Авто-извлечение графа (LLM + NER)
- Repomix-индексация codebase с sibling-связями
- BFS обход графа с мета-фильтром
- Графовая визуализация (vis.js)
- Docker (Qdrant + Postgres)
- Миграция с ChromaDB

---

## 2. Сравнение с конкурентами

### 2.1 Microsoft GraphRAG
| Характеристика | Microsoft GraphRAG | graphrag |
|---|---|---|
| **Звёзды GitHub** | 34.3k | — |
| **Архитектура** | LLM community detection + map-reduce | RRF hybrid search + граф |
| **MCP транспорт** | Нет | **Да (stdio + SSE)** |
| **Эмбеддинги** | OpenAI GPT | BGE-M3 / Ollama / OpenRouter |
| **Стоимость индексации** | Высокая | Низкая |
| **Community reports** | Да | Нет |
| **Визуализация** | Нет | Да (vis.js) |
| **Сложность кода** | Высокая (тысячи строк) | Умеренная |

**graphrag преимущества**: MCP, низкая стоимость, визуализация, несколько провайдеров эмбеддингов, language-aware поиск

### 2.2 LightRAG (HKUDS)
| Характеристика | LightRAG | graphrag |
|---|---|---|
| **Звёзды GitHub** | 37.5k | — |
| **Архитектура** | Dual-level KG + vector | RRF hybrid + граф |
| **MCP транспорт** | Нет | **Да** |
| **WebUI** | **Да** | Нет |
| **Multimodal** | **Да (MinerU/Docling)** | Нет |
| **Режимы запросов** | 5 (local/global/hybrid/naive/mix) | semantic/BM25/hybrid |
| **Storage backends** | PostgreSQL, MongoDB, Neo4j, OpenSearch | Qdrant + SQLite/PG |
| **Reranker** | Да | Да |
| **Incremental update** | **Да** | Нет |

**graphrag преимущества**: MCP транспорт, легче, BGE-M3, лучше для русскоязычных проектов (language-aware alpha)

### 2.3 nano-graphrag
| Характеристика | nano-graphrag | graphrag |
|---|---|---|
| **Звёзды GitHub** | 3.9k | — |
| **Размер кода** | ~1100 LOC | ~5000+ LOC |
| **MCP** | Нет | **Да** |
| **BM25** | Нет | **Да** |
| **Reranker** | Нет | **Да** |
| **Query expansion** | Нет | **Да** |
| **HTTP API** | Нет | **Да (FastAPI)** |
| **Визуализация** | GraphML | vis.js |

**graphrag преимущества**: Значительно богаче фичами, production-ready транспорты

### 2.4 fast-graphrag (CircleMind)
| Характеристика | fast-graphrag | graphrag |
|---|---|---|
| **Звёзды GitHub** | 3.8k | — |
| **MCP** | Нет | **Да** |
| **Open Source** | Да (с managed service) | **Полностью open-source** |
| **PageRank** | **Да** | Нет |
| **Стоимость** | $0.08 vs MS $0.48 | Сравнимая |
| **HTTP API** | Нет | **Да (FastAPI)** |

**graphrag преимущества**: Полностью open-source, MCP, больше транспортов, HTTP API

### 2.5 MiniRAG (HKUDS)
| Характеристика | MiniRAG | graphrag |
|---|---|---|
| **Звёзды GitHub** | 2k | — |
| **Целевые модели** | SLM (Phi-3, 1.5B-4B) | Любые (через Ollama/OpenAI) |
| **MCP** | Нет | **Да** |
| **SLM-оптимизация** | **Да (HGI)** | Нет |
| **On-device** | **Да** | Нет |
| **Фичи** | Базовые | **Богаче** |

**graphrag преимущества**: Больше фич, мощнее модели, MCP; MiniRAG лучше для edge

### 2.6 LlamaIndex
| Характеристика | LlamaIndex | graphrag |
|---|---|---|
| **Звёзды GitHub** | 50.8k | — |
| **Тип** | Фреймворк | **Готовый сервер** |
| **Knowledge Graph** | Одна из 300+ фич | **Основная специализация** |
| **MCP** | Нет | **Да** |
| **Интеграции** | **300+** | 5-10 |
| **LlamaParse** | **Да** | Нет |

**graphrag преимущества**: Готовый сервер с MCP, специализация на графовом RAG

### 2.7 LangChain
| Характеристика | LangChain | graphrag |
|---|---|---|
| **Звёзды GitHub** | 141k | — |
| **Тип** | Фреймворк | **Готовый сервер** |
| **GraphRAG специализация** | Нет (через Neo4j) | **Да** |
| **MCP** | Нет (есть .mcp.json) | **Да (полноценный)** |
| **Экосистема** | **Огромная (LangSmith)** | Минимальная |

**graphrag преимущества**: Готовое решение под ключ, MCP, специализация

### 2.8 HippoRAG
| Характеристика | HippoRAG | graphrag |
|---|---|---|
| **Архитектура** | PPR-based retrieval | RRF + граф + reranker |
| **MCP** | Нет | **Да** |
| **Фичи** | Базовые | **Богаче** |
| **Активность** | Низкая | Высокая |

---

## 3. SWOT-анализ

### Сильные стороны (Strengths)
1. **MCP протокол** — единственный GraphRAG с полноценным MCP (stdio + SSE)
2. **Гибридный поиск** — RRF + BM25 + reranker + query expansion
3. **Language-aware alpha** — оптимально для русскоязычных проектов
4. **Множество провайдеров эмбеддингов** — BGE-M3 (локально, бесплатно), Ollama, OpenRouter
5. **Авто-извлечение графа** — LLM + NER
6. **Repomix индексация** — для codebase
7. **Два бэкенда** — SQLite (dev) и PostgreSQL (prod)
8. **Docker-ready**
9. **Визуализация графа** — vis.js

### Слабые стороны (Weaknesses)
1. **Версия 0.2.0** — ранняя стадия зрелости
2. **Нет WebUI** — только CLI/HTTP/MCP (добавлен в PLAN.md как Фаза 3.5)
3. **Нет community reports** — как в MS GraphRAG
4. **Ограниченные storage backend** — только Qdrant + SQLite/PG
5. **Нет multimodal** — изображения, таблицы, PDF
6. **Нет инкрементального обновления графа**
7. **Нет multi-tenancy**
8. **Нет тестов производительности** (benchmarks)

### Возможности (Opportunities)
1. MCP протокол набирает популярность (Claude, Cline, IDE)
2. Рынок GraphRAG активно растёт
3. Спрос на privacy-safe, on-premise RAG
4. Русскоязычная ниша (language-aware alpha)
5. Интеграция с IDE через MCP

### Угрозы (Threats)
1. LightRAG (37.5k stars) доминирует с огромным комьюнити
2. Microsoft GraphRAG (34.3k stars) — бренд и research
3. LlamaIndex и LangChain внедряют GraphRAG как фичу
4. Конкуренты быстро развивают MCP поддержку

---

## 4. Рекомендации по улучшению

### Критические (для конкурентоспособности)
1. **Incremental graph updates** — без этого проигрыш LightRAG
2. **WebUI** — для non-technical пользователей
3. **CI/CD бенчмарки** — NDCG@k, latency, throughput
4. **Multi-tenancy** — для enterprise

### Важные
5. **Поддержка Neo4j как graph backend**
6. **Community detection / reports** (алгоритмы Лувена)
7. **Multimodal** (PDF, изображения — через OCR/vision)
8. **Больше storage backends** (PostgreSQL vector, Redis, MongoDB)

### Дополнительные
9. **Публичный roadmap** (GitHub Projects)
10. **Документация на английском** для global reach
11. **PyPI публикация** (`pip install graphrag`)
12. **Примеры интеграции** с Claude, Cline, VS Code

---

## 5. Вывод

**graphrag занимает уникальную нишу:** единственный open-source GraphRAG-сервер с:
- Полноценным MCP протоколом (stdio + SSE)
- Гибридным поиском (dense + sparse + graph)
- Reranker + Query expansion
- Language-aware поиском (кириллица)

**Основной конкурент — LightRAG**, у которого больше звёзд, фич и сообщества, но нет MCP. graphrag может выиграть за счёт:
1) фокуса на MCP экосистему
2) русского языка
3) лёгкости и низкой стоимости индексации
4) on-premise развёртывания

**Риски**: LightRAG и Microsoft быстро догонят по MCP. Нужно срочно развивать incremental updates, WebUI и multi-tenancy.
