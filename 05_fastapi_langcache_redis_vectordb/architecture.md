# Architecture Diagram

```mermaid
graph TB
    subgraph Client
        C[Client Application]
        RC[Rich Console UI]
    end

    subgraph FastAPI_Server
        FA[FastAPI App]
        SC[Semantic Cache Service]
        ST[Sentence Transformer]
    end

    subgraph Redis_Vector_DB
        RDB[(Redis DB)]
        RSI[RediSearch Index]
        HNSW[HNSW Index]
    end

    %% Client to Server interactions
    C -->|1. HTTP Request| FA
    FA -->|8. HTTP Response| C
    C -->|9. Display Results| RC

    %% Server internal flow
    FA -->|2. Query| SC
    SC -->|3. Generate Embedding| ST
    ST -->|4. Embedding Vector| SC
    
    %% Cache operations
    SC -->|5. Vector Search| RSI
    RSI -->|6. Query HNSW| HNSW
    HNSW -->|7a. Cache Hit| SC
    SC -->|7b. Cache Miss: Store| RDB

    %% Data structures
    subgraph Data_Structures
        QR[Query Request<br/>- query: str<br/>- use_cache: bool<br/>- similarity_threshold: float]
        CR[Cache Response<br/>- query: str<br/>- response: str<br/>- cache_hit: bool<br/>- similarity_score: float]
    end

    %% Redis index structure
    subgraph Redis_Schema
        IS[Index Schema<br/>- embedding: HNSW Vector<br/>- query: Text<br/>- response: Text<br/>- timestamp: Number]
    end

    style FastAPI_Server fill:#f9f,stroke:#333,stroke-width:2px
    style Redis_Vector_DB fill:#bbf,stroke:#333,stroke-width:2px
    style Data_Structures fill:#bfb,stroke:#333,stroke-width:2px
    style Redis_Schema fill:#fbb,stroke:#333,stroke-width:2px

    classDef component fill:#f9f,stroke:#333,stroke-width:2px;
    classDef database fill:#bbf,stroke:#333,stroke-width:2px;
    classDef ui fill:#bfb,stroke:#333,stroke-width:2px;

    class FA,SC,ST component;
    class RDB,RSI,HNSW database;
    class C,RC ui;
```

## Flow Description

1. **Client Request**: Client sends a query with optional caching parameters
2. **Query Processing**: FastAPI server receives request and forwards to Semantic Cache Service
3. **Embedding Generation**: Query is converted to vector embedding using Sentence Transformer
4. **Vector Search**: 
   - System searches Redis for similar queries using HNSW index
   - Similarity threshold determines cache hit/miss
5. **Cache Operations**:
   - Hit: Returns cached response with similarity score
   - Miss: Generates new response and stores in cache
6. **Response**: Returns results to client with cache status
7. **Display**: Client displays results in formatted table

## Key Components

- **FastAPI Server**: Main application server
- **Semantic Cache Service**: Manages caching logic and vector operations
- **Redis Vector DB**: 
  - Stores embeddings and responses
  - Uses HNSW index for efficient similarity search
  - Maintains cache size limits
- **Client**: 
  - Sends queries
  - Displays results with Rich console formatting
  - Shows cache statistics 