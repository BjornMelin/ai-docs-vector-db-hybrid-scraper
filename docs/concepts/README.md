# Concepts

> **Purpose**: Understanding-oriented explanations of system design  
> **Audience**: Developers wanting to understand the why and how

## Concept Categories

### 🏗️ Architecture
System design and component relationships:
- [**System Overview**](./architecture/system-overview.md) - High-level architecture and data flow
- [**V1 Architecture**](./architecture/v1-architecture.md) - Detailed component synergy
- [**Browser Architecture**](./architecture/browser-architecture.md) - 5-tier scraping system design
- [**Scraping Architecture**](../concepts/architecture/scraping-architecture.md) - Web scraping patterns
- [**Client Management**](../concepts/architecture/client-management.md) - Singleton patterns and lifecycle

### 🧠 Features
Deep dives into core functionality:
- [**Chunking Theory**](./features/chunking-theory.md) - AST-based document chunking explained

## Understanding the System

### Core Principles

The system is built on these foundational concepts:

1. **Hybrid Search** - Combines semantic and keyword search
2. **Tiered Automation** - 5-tier browser automation system
3. **Intelligent Caching** - Multi-layer caching strategy
4. **Modular Design** - Composable, testable components

### Architecture Philosophy

- **Performance First** - Sub-100ms search latency
- **Scalability** - Horizontal scaling support
- **Reliability** - Graceful failure handling
- **Developer Experience** - Clear APIs and documentation

## Learning Path

1. **Start with**: [System Overview](./architecture/system-overview.md) - Get the big picture
2. **Dive into**: [V1 Architecture](./architecture/v1-architecture.md) - Understand components
3. **Explore**: [Browser Architecture](./architecture/browser-architecture.md) - Learn scraping design
4. **Study**: [Chunking Theory](./features/chunking-theory.md) - Deep dive into features

## Design Decisions

Key architectural decisions explained:
- **Why Qdrant?** - Vector database choice reasoning
- **Why 5-tier scraping?** - Balancing performance and capability
- **Why FastAPI?** - Framework selection criteria
- **Why Pydantic?** - Type safety and validation

## Related Documentation

- 🛠️ [How-to Guides](../how-to-guides/) - Practical implementation
- 📋 [Reference](../reference/) - Technical specifications
- 🚀 [Tutorials](../tutorials/) - Hands-on learning