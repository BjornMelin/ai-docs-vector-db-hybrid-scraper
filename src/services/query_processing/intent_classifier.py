"""Advanced Query Intent Classification.

This module provides comprehensive query intent classification using both
rule-based patterns and semantic analysis to categorize queries into 14
distinct intent categories for optimal search strategy selection.
"""

import logging
import re
from typing import Any

import numpy as np

from .models import QueryComplexity
from .models import QueryIntent
from .models import QueryIntentClassification

logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """Advanced query intent classifier using hybrid rule-based and semantic analysis.

    Expands classification from 4 basic categories to 14 comprehensive categories
    for intelligent search strategy selection and query processing optimization.
    """

    def __init__(self, embedding_manager: Any = None):
        """Initialize the advanced query intent classifier.

        Args:
            embedding_manager: Optional EmbeddingManager for semantic analysis
        """
        self.embedding_manager = embedding_manager
        self._initialized = False

        # Classification patterns for each of the 14 intent categories
        self._intent_patterns = {
            # Basic categories (existing)
            QueryIntent.CONCEPTUAL: {
                "keywords": [
                    "what is",
                    "what are",
                    "explain",
                    "definition",
                    "concept",
                    "understand",
                    "overview",
                    "introduction",
                    "fundamentals",
                    "theory",
                    "principle",
                    "basic",
                    "meaning",
                    "purpose",
                ],
                "patterns": [
                    r"what\s+(is|are|does|means?)",
                    r"explain\s+(?:what|how|why)",
                    r"(?:give\s+)?(?:an?\s+)?(?:overview|introduction|explanation)",
                    r"help\s+(?:me\s+)?understand",
                    r"(?:basic|fundamental)\s+(?:concept|principle)",
                ],
                "weight": 1.0,
            },
            QueryIntent.PROCEDURAL: {
                "keywords": [
                    "how to",
                    "step by step",
                    "tutorial",
                    "guide",
                    "walkthrough",
                    "instructions",
                    "process",
                    "procedure",
                    "implement",
                    "create",
                    "build",
                    "setup",
                    "configure",
                    "install",
                ],
                "patterns": [
                    r"how\s+(?:do\s+i|to|can\s+i)",
                    r"step\s+by\s+step",
                    r"(?:show\s+me\s+)?how\s+to",
                    r"(?:tutorial|guide|walkthrough)\s+(?:on|for)",
                    r"(?:setup|configure|install|implement|create|build)",
                ],
                "weight": 1.0,
            },
            QueryIntent.FACTUAL: {
                "keywords": [
                    "when",
                    "where",
                    "who",
                    "which",
                    "version",
                    "latest",
                    "current",
                    "support",
                    "compatible",
                    "requirements",
                    "specifications",
                    "details",
                ],
                "patterns": [
                    r"(?:when|where|who|which)\s+",
                    r"(?:what\s+)?version",
                    r"(?:latest|current|newest)\s+version",
                    r"(?:system\s+)?requirements",
                    r"(?:technical\s+)?specifications",
                    r"(?:is\s+)?(?:supported|compatible)",
                ],
                "weight": 1.0,
            },
            QueryIntent.TROUBLESHOOTING: {
                "keywords": [
                    "error",
                    "problem",
                    "issue",
                    "bug",
                    "broken",
                    "not working",
                    "failed",
                    "crash",
                    "exception",
                    "fix",
                    "solve",
                    "resolve",
                    "debug",
                    "troubleshoot",
                    "help",
                ],
                "patterns": [
                    r"(?:getting|receiving)\s+(?:an?\s+)?error",
                    r"(?:not\s+)?working\s+(?:properly|correctly)?",
                    r"(?:having\s+)?(?:trouble|issues?|problems?)",
                    r"(?:how\s+to\s+)?(?:fix|solve|resolve|debug)",
                    r"(?:keeps?\s+)?(?:crashing|failing)",
                    r"exception\s+(?:thrown|occurred)",
                ],
                "weight": 1.0,
            },
            # Advanced categories (new)
            QueryIntent.COMPARATIVE: {
                "keywords": [
                    "vs",
                    "versus",
                    "compare",
                    "comparison",
                    "difference",
                    "differences",
                    "better",
                    "best",
                    "alternative",
                    "alternatives",
                    "choose",
                    "choice",
                    "pros and cons",
                    "advantages",
                    "disadvantages",
                ],
                "patterns": [
                    r"\bvs\b|\bversus\b",
                    r"(?:compare|comparison)\s+(?:between|of)",
                    r"(?:what\s+(?:is\s+)?)?(?:difference|differences)\s+between",
                    r"(?:which\s+is\s+)?(?:better|best)",
                    r"(?:pros\s+and\s+cons|advantages?\s+and\s+disadvantages?)",
                    r"(?:should\s+i\s+)?(?:choose|use|pick)",
                ],
                "weight": 1.2,
            },
            QueryIntent.ARCHITECTURAL: {
                "keywords": [
                    "architecture",
                    "design",
                    "pattern",
                    "patterns",
                    "structure",
                    "system design",
                    "microservices",
                    "monolith",
                    "scalability",
                    "design pattern",
                    "architecture pattern",
                    "system architecture",
                ],
                "patterns": [
                    r"(?:system\s+)?(?:architecture|design)",
                    r"(?:design\s+)?patterns?",
                    r"(?:architectural\s+)?(?:decisions?|choices?)",
                    r"(?:microservices?|monolithic?)\s+(?:architecture|design)",
                    r"(?:scalable|scalability)\s+(?:architecture|design)",
                    r"(?:how\s+to\s+)?(?:architect|structure)\s+",
                ],
                "weight": 1.3,
            },
            QueryIntent.PERFORMANCE: {
                "keywords": [
                    "performance",
                    "optimization",
                    "optimize",
                    "speed",
                    "fast",
                    "slow",
                    "latency",
                    "throughput",
                    "scaling",
                    "memory",
                    "cpu",
                    "efficiency",
                    "benchmark",
                    "profiling",
                    "bottleneck",
                ],
                "patterns": [
                    r"(?:performance|speed)\s+(?:optimization|tuning|improvement)",
                    r"(?:optimize|improve)\s+(?:performance|speed)",
                    r"(?:reduce|decrease)\s+(?:latency|response\s+time)",
                    r"(?:increase|improve)\s+(?:throughput|efficiency)",
                    r"(?:memory|cpu)\s+(?:usage|optimization|efficiency)",
                    r"(?:profiling|benchmarking|bottleneck)",
                ],
                "weight": 1.2,
            },
            QueryIntent.SECURITY: {
                "keywords": [
                    "security",
                    "secure",
                    "authentication",
                    "authorization",
                    "encryption",
                    "vulnerability",
                    "attack",
                    "protection",
                    "firewall",
                    "ssl",
                    "tls",
                    "certificate",
                    "oauth",
                    "jwt",
                    "permissions",
                    "access control",
                ],
                "patterns": [
                    r"(?:security|secure)\s+(?:implementation|setup|configuration)",
                    r"(?:authentication|authorization)\s+(?:setup|configuration|flow)",
                    r"(?:ssl|tls)\s+(?:setup|configuration|certificate)",
                    r"(?:oauth|jwt)\s+(?:implementation|setup|token)",
                    r"(?:access\s+control|permissions?)\s+(?:setup|management)",
                    r"(?:vulnerability|security)\s+(?:assessment|audit)",
                ],
                "weight": 1.3,
            },
            QueryIntent.INTEGRATION: {
                "keywords": [
                    "integration",
                    "integrate",
                    "api",
                    "webhook",
                    "rest",
                    "graphql",
                    "third party",
                    "external",
                    "connect",
                    "connection",
                    "endpoint",
                    "sdk",
                    "library",
                    "plugin",
                    "middleware",
                ],
                "patterns": [
                    r"(?:api\s+)?integration\s+(?:with|to)",
                    r"(?:integrate|connect)\s+(?:with|to)",
                    r"(?:third\s+party|external)\s+(?:api|service|integration)",
                    r"(?:rest|graphql)\s+(?:api|integration|endpoint)",
                    r"(?:webhook|callback)\s+(?:setup|integration|configuration)",
                    r"(?:sdk|library|plugin)\s+(?:integration|usage)",
                ],
                "weight": 1.2,
            },
            QueryIntent.BEST_PRACTICES: {
                "keywords": [
                    "best practices",
                    "best practice",
                    "recommended",
                    "convention",
                    "conventions",
                    "guidelines",
                    "standards",
                    "coding standards",
                    "industry standard",
                    "good practice",
                    "proper way",
                ],
                "patterns": [
                    r"best\s+practices?",
                    r"(?:recommended|proper)\s+(?:way|approach|method)",
                    r"(?:coding|programming)\s+(?:standards?|conventions?|guidelines?)",
                    r"(?:industry\s+)?(?:standards?|conventions?)",
                    r"(?:good|proper)\s+practices?",
                    r"(?:what\s+is\s+)?(?:the\s+)?(?:recommended|standard)\s+way",
                ],
                "weight": 1.1,
            },
            QueryIntent.CODE_REVIEW: {
                "keywords": [
                    "code review",
                    "review",
                    "feedback",
                    "improve",
                    "refactor",
                    "clean code",
                    "code quality",
                    "optimization",
                    "suggestions",
                    "better way",
                    "code analysis",
                    "static analysis",
                ],
                "patterns": [
                    r"code\s+(?:review|analysis|quality|improvement)",
                    r"(?:review|analyze)\s+(?:my\s+)?code",
                    r"(?:how\s+to\s+)?(?:improve|refactor|optimize)\s+(?:this\s+)?code",
                    r"(?:better|cleaner)\s+way\s+to\s+(?:write|implement)",
                    r"(?:static\s+)?(?:code\s+)?analysis",
                    r"(?:suggestions?|feedback)\s+(?:on\s+)?(?:my\s+)?code",
                ],
                "weight": 1.2,
            },
            QueryIntent.MIGRATION: {
                "keywords": [
                    "migration",
                    "migrate",
                    "upgrade",
                    "update",
                    "transition",
                    "move from",
                    "convert",
                    "port",
                    "legacy",
                    "modernize",
                    "version upgrade",
                    "framework migration",
                ],
                "patterns": [
                    r"(?:migrate|migration)\s+(?:from|to)",
                    r"(?:upgrade|update)\s+(?:from|to)",
                    r"(?:move|transition|convert)\s+from\s+.+\s+to",
                    r"(?:legacy\s+)?(?:system\s+)?(?:migration|modernization)",
                    r"(?:framework|library|platform)\s+(?:migration|transition)",
                    r"(?:version\s+)?(?:upgrade|update)\s+(?:guide|process)",
                ],
                "weight": 1.3,
            },
            QueryIntent.DEBUGGING: {
                "keywords": [
                    "debug",
                    "debugging",
                    "debugger",
                    "breakpoint",
                    "stack trace",
                    "error message",
                    "exception",
                    "logging",
                    "trace",
                    "step through",
                    "inspect",
                    "console",
                    "dev tools",
                ],
                "patterns": [
                    r"(?:how\s+to\s+)?debug(?:ging)?",
                    r"(?:set\s+)?breakpoints?",
                    r"(?:stack\s+)?trace\s+(?:analysis|interpretation)",
                    r"(?:error\s+)?(?:message|exception)\s+(?:meaning|explanation)",
                    r"(?:logging|console)\s+(?:setup|configuration|output)",
                    r"(?:step\s+through|inspect)\s+(?:code|execution)",
                ],
                "weight": 1.2,
            },
            QueryIntent.CONFIGURATION: {
                "keywords": [
                    "configuration",
                    "configure",
                    "config",
                    "setup",
                    "settings",
                    "environment",
                    "variables",
                    "parameters",
                    "options",
                    "customize",
                    "initialization",
                    "setup guide",
                ],
                "patterns": [
                    r"(?:how\s+to\s+)?(?:configure|setup|initialize)",
                    r"(?:configuration|config)\s+(?:file|settings?|options?)",
                    r"(?:environment\s+)?(?:variables?|settings?|configuration)",
                    r"(?:setup|installation)\s+(?:guide|process|instructions)",
                    r"(?:customize|modify)\s+(?:settings?|configuration)",
                    r"(?:default\s+)?(?:configuration|settings?|parameters?)",
                ],
                "weight": 1.1,
            },
        }

        # Complexity indicators for determining query complexity
        self._complexity_indicators = {
            QueryComplexity.SIMPLE: {
                "patterns": [r"^what\s+is", r"^how\s+to", r"^where\s+", r"^when\s+"],
                "max_words": 6,
                "max_tech_terms": 1,
            },
            QueryComplexity.MODERATE: {
                "patterns": [r"compare|vs|versus", r"best\s+practice", r"how\s+do\s+i"],
                "max_words": 12,
                "max_tech_terms": 3,
            },
            QueryComplexity.COMPLEX: {
                "patterns": [
                    r"architecture|design\s+pattern",
                    r"integration.*with",
                    r"migration.*from",
                ],
                "max_words": 20,
                "max_tech_terms": 5,
            },
            QueryComplexity.EXPERT: {
                "patterns": [
                    r"(?:performance|security).*optimization",
                    r"scalable.*architecture",
                ],
                "max_words": 50,
                "max_tech_terms": 10,
            },
        }

        # Technical domain terms for domain detection
        self._domain_terms = {
            "web_development": [
                "html",
                "css",
                "javascript",
                "react",
                "vue",
                "angular",
                "dom",
                "frontend",
            ],
            "backend": [
                "api",
                "rest",
                "graphql",
                "database",
                "server",
                "microservices",
                "backend",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "ci/cd",
                "deployment",
                "infrastructure",
                "cloud",
            ],
            "data_science": [
                "machine learning",
                "ai",
                "data",
                "analysis",
                "python",
                "pandas",
                "numpy",
            ],
            "mobile": [
                "ios",
                "android",
                "mobile",
                "app",
                "native",
                "flutter",
                "react native",
            ],
            "security": [
                "security",
                "encryption",
                "authentication",
                "authorization",
                "ssl",
                "oauth",
            ],
            "database": [
                "sql",
                "nosql",
                "mongodb",
                "postgresql",
                "mysql",
                "database",
                "query",
            ],
        }

    async def initialize(self) -> None:
        """Initialize the intent classifier."""
        self._initialized = True
        logger.info(
            "Advanced QueryIntentClassifier initialized with 14 intent categories"
        )

    async def classify_query_advanced(
        self, query: str, context: dict[str, Any] | None = None
    ) -> QueryIntentClassification:
        """Perform advanced multi-label query intent classification.

        Args:
            query: User query to classify
            context: Optional context information

        Returns:
            QueryIntentClassification: Classification results with confidence scores

        Raises:
            RuntimeError: If classifier not initialized
        """
        if not self._initialized:
            raise RuntimeError("QueryIntentClassifier not initialized")

        if not query.strip():
            return QueryIntentClassification(
                primary_intent=QueryIntent.FACTUAL,
                secondary_intents=[],
                confidence_scores={QueryIntent.FACTUAL: 0.1},
                complexity_level=QueryComplexity.SIMPLE,
                classification_reasoning="Empty query provided",
            )

        # Normalize query
        query_lower = query.lower().strip()

        # Calculate intent scores using rule-based patterns
        intent_scores = await self._calculate_intent_scores(query_lower)

        # Add semantic analysis if available
        if self.embedding_manager:
            try:
                semantic_scores = await self._semantic_intent_classification(query)
                # Blend semantic and rule-based scores (70% rule-based, 30% semantic)
                for intent in intent_scores:
                    if intent in semantic_scores:
                        intent_scores[intent] = (
                            intent_scores[intent] * 0.7 + semantic_scores[intent] * 0.3
                        )
            except Exception as e:
                logger.warning(f"Semantic intent classification failed: {e}")

        # Determine complexity level
        complexity_level = self._assess_complexity(query_lower)

        # Detect domain category
        domain_category = self._detect_domain(query_lower)

        # Sort intents by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        # Determine primary and secondary intents
        if not sorted_intents or sorted_intents[0][1] < 0.1:
            primary_intent = QueryIntent.FACTUAL  # Default fallback
            secondary_intents = []
        else:
            primary_intent = sorted_intents[0][0]
            # Include secondary intents with score > 0.2 and within 0.4 of primary
            secondary_intents = [
                intent
                for intent, score in sorted_intents[1:3]
                if score > 0.2 and (sorted_intents[0][1] - score) < 0.4
            ]

        # Generate reasoning
        reasoning_parts = []
        for intent, score in sorted_intents[:3]:
            if score > 0.1:
                reasoning_parts.append(f"{intent.value}:{score:.2f}")
        reasoning = f"Rule-based + semantic analysis: {', '.join(reasoning_parts)}"

        # Generate suggested follow-ups based on intent
        suggested_followups = self._generate_followups(primary_intent, query)

        # Determine if additional context is needed
        requires_context = self._requires_context(
            primary_intent, complexity_level, query_lower
        )

        return QueryIntentClassification(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence_scores=intent_scores,
            complexity_level=complexity_level,
            domain_category=domain_category,
            classification_reasoning=reasoning,
            requires_context=requires_context,
            suggested_followups=suggested_followups,
        )

    async def _calculate_intent_scores(self, query: str) -> dict[QueryIntent, float]:
        """Calculate intent scores using rule-based pattern matching."""
        scores = dict.fromkeys(QueryIntent, 0.0)

        for intent, config in self._intent_patterns.items():
            score = 0.0

            # Keyword matching
            keyword_matches = sum(
                1 for keyword in config["keywords"] if keyword in query
            )
            if keyword_matches > 0:
                score += min(keyword_matches / len(config["keywords"]), 0.6) * 0.6

            # Pattern matching
            pattern_matches = sum(
                1
                for pattern in config["patterns"]
                if re.search(pattern, query, re.IGNORECASE)
            )
            if pattern_matches > 0:
                score += min(pattern_matches / len(config["patterns"]), 0.8) * 0.8

            # Apply intent weight
            score *= config.get("weight", 1.0)

            # Ensure score doesn't exceed 1.0
            scores[intent] = min(score, 1.0)

        return scores

    async def _semantic_intent_classification(
        self, query: str
    ) -> dict[QueryIntent, float]:
        """Perform semantic classification using embeddings."""
        # Reference queries for each intent category
        reference_queries = {
            QueryIntent.CONCEPTUAL: "What is the concept and how does it work?",
            QueryIntent.PROCEDURAL: "How do I implement this step by step?",
            QueryIntent.FACTUAL: "What are the specifications and requirements?",
            QueryIntent.TROUBLESHOOTING: "How to fix this error and resolve the problem?",
            QueryIntent.COMPARATIVE: "What are the differences and which is better?",
            QueryIntent.ARCHITECTURAL: "How to design the system architecture?",
            QueryIntent.PERFORMANCE: "How to optimize performance and improve speed?",
            QueryIntent.SECURITY: "How to implement security and authentication?",
            QueryIntent.INTEGRATION: "How to integrate with external APIs?",
            QueryIntent.BEST_PRACTICES: "What are the recommended best practices?",
            QueryIntent.CODE_REVIEW: "How to improve and refactor this code?",
            QueryIntent.MIGRATION: "How to migrate and upgrade the system?",
            QueryIntent.DEBUGGING: "How to debug and trace the execution?",
            QueryIntent.CONFIGURATION: "How to configure and setup the system?",
        }

        try:
            # Get embeddings for query and reference texts
            all_texts = [query] + list(reference_queries.values())

            result = await self.embedding_manager.generate_embeddings(
                texts=all_texts,
                quality_tier=None,
                auto_select=True,
            )

            if not result.get("success", False) or not result.get("embeddings"):
                return {}

            embeddings = result["embeddings"]
            
            # Defensive check: ensure we have the expected number of embeddings
            expected_count = len(reference_queries) + 1  # +1 for the query
            if len(embeddings) != expected_count:
                logger.error(
                    f"Expected {expected_count} embeddings but got {len(embeddings)}"
                )
                return {}
            
            query_embedding = embeddings[0]
            reference_embeddings = embeddings[1:]

            # Calculate cosine similarity scores
            scores = {}
            for i, (intent, _) in enumerate(reference_queries.items()):
                if i >= len(reference_embeddings):
                    logger.error(
                        f"Index {i} out of range for reference_embeddings of length {len(reference_embeddings)}"
                    )
                    break
                similarity = self._cosine_similarity(
                    query_embedding, reference_embeddings[i]
                )
                scores[intent] = max(0.0, similarity)  # Ensure non-negative

            return scores

        except Exception as e:
            logger.error(f"Semantic intent classification failed: {e}")
            return {}

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            magnitude1 = np.linalg.norm(vec1_np)
            magnitude2 = np.linalg.norm(vec2_np)

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return float(dot_product / (magnitude1 * magnitude2))
        except Exception:
            return 0.0

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on length, patterns, and technical terms."""
        word_count = len(query.split())

        # Count technical terms
        tech_term_count = 0
        for domain_terms in self._domain_terms.values():
            tech_term_count += sum(1 for term in domain_terms if term in query)

        # Check complexity patterns in reverse order (expert first)
        for complexity in reversed(list(QueryComplexity)):
            indicators = self._complexity_indicators[complexity]

            # Check patterns
            pattern_match = any(
                re.search(pattern, query, re.IGNORECASE)
                for pattern in indicators["patterns"]
            )

            # Check word count and tech terms
            word_match = word_count <= indicators["max_words"]
            tech_match = tech_term_count <= indicators["max_tech_terms"]

            if (
                pattern_match
                or (not word_match and tech_term_count > 3)
                or (word_match and tech_match)
            ):
                return complexity

        return QueryComplexity.SIMPLE

    def _detect_domain(self, query: str) -> str | None:
        """Detect the technical domain/category of the query."""
        domain_scores = {}

        for domain, terms in self._domain_terms.items():
            score = sum(1 for term in terms if term in query)
            if score > 0:
                domain_scores[domain] = score / len(terms)

        if not domain_scores:
            return None

        return max(domain_scores, key=domain_scores.get)

    def _generate_followups(self, intent: QueryIntent, query: str) -> list[str]:
        """Generate suggested follow-up questions based on intent."""
        followup_templates = {
            QueryIntent.CONCEPTUAL: [
                "What are the key benefits and use cases?",
                "How does this compare to alternatives?",
                "What are the prerequisites to understand this?",
            ],
            QueryIntent.PROCEDURAL: [
                "What are the common pitfalls to avoid?",
                "Are there any alternative approaches?",
                "What tools or dependencies are needed?",
            ],
            QueryIntent.TROUBLESHOOTING: [
                "What are the common causes of this issue?",
                "How can I prevent this in the future?",
                "Are there any diagnostic tools to help?",
            ],
            QueryIntent.PERFORMANCE: [
                "What are the key performance metrics to monitor?",
                "How can I benchmark the improvements?",
                "What are the trade-offs of different optimizations?",
            ],
            QueryIntent.SECURITY: [
                "What are the current security best practices?",
                "How can I test the security implementation?",
                "What are common security vulnerabilities to avoid?",
            ],
        }

        return followup_templates.get(
            intent,
            [
                "Can you provide more specific details?",
                "What is your current setup or context?",
                "Are there any constraints or requirements to consider?",
            ],
        )

    def _requires_context(
        self, intent: QueryIntent, complexity: QueryComplexity, query: str
    ) -> bool:
        """Determine if the query requires additional context for proper handling."""
        # High complexity queries usually need more context
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            return True

        # Certain intents benefit from context
        context_heavy_intents = {
            QueryIntent.ARCHITECTURAL,
            QueryIntent.INTEGRATION,
            QueryIntent.MIGRATION,
            QueryIntent.TROUBLESHOOTING,
            QueryIntent.CODE_REVIEW,
        }

        if intent in context_heavy_intents:
            return True

        # Vague or very short queries need context
        if len(query.split()) < 4:
            return True

        return False

    async def cleanup(self) -> None:
        """Cleanup classifier resources."""
        self._initialized = False
        logger.info("QueryIntentClassifier cleaned up")
