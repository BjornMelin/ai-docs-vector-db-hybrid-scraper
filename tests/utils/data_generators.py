"""Data generators for testing using Hypothesis and custom patterns.

This module provides comprehensive data generation utilities for creating test data
that covers edge cases, realistic scenarios, and property-based testing patterns.
"""

import random
import string
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any


try:
    import hypothesis.strategies as st
    from hypothesis import given, settings

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

try:
    from faker import Faker

    fake = Faker()
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

    # Simple fallback for name generation
    class SimpleFaker:
        @staticmethod
        def name():
            names = [
                "John Doe",
                "Jane Smith",
                "Bob Johnson",
                "Alice Brown",
                "Charlie Wilson",
            ]
            return random.choice(names)

        @staticmethod
        def url():
            domains = ["example.com", "test.org", "demo.net"]
            return f"https://{random.choice(domains)}/page{random.randint(1, 100)}"

        @staticmethod
        def sentence(nb_words=6):
            words = [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "lazy",
                "dog",
                "test",
                "data",
            ]
            return " ".join(random.choices(words, k=nb_words)).capitalize() + "."

        @staticmethod
        def paragraph(nb_sentences=4):
            return " ".join([SimpleFaker.sentence() for _ in range(nb_sentences)])

        @staticmethod
        def word():
            words = [
                "test",
                "data",
                "sample",
                "example",
                "demo",
                "mock",
                "stub",
                "fake",
            ]
            return random.choice(words)

        @staticmethod
        def date_time_between(start_date, end_date):
            # Simple date generation
            start = datetime.now(tz=UTC) - timedelta(days=365)
            end = datetime.now(tz=UTC)
            return start + timedelta(
                seconds=random.randint(0, int((end - start).total_seconds()))
            )

        @staticmethod
        def image_url():
            return f"https://via.placeholder.com/{random.randint(100, 800)}x{random.randint(100, 600)}"

    fake = SimpleFaker()


class HypothesisStrategies:
    """Collection of Hypothesis strategies for property-based testing.

    Note: This class provides both Hypothesis-based strategies (when available)
    and fallback implementations for when Hypothesis is not installed.
    """

    @staticmethod
    def document_id():
        """Generate valid document IDs."""
        id_types = ["uuid", "custom", "numeric"]
        id_type = random.choice(id_types)

        if id_type == "uuid":
            return str(uuid.uuid4())
        elif id_type == "numeric":
            return str(random.randint(1, 999999))
        else:
            # Custom alphanumeric ID
            length = random.randint(8, 32)
            chars = string.ascii_letters + string.digits + "_-"
            return "".join(random.choices(chars, k=length))

    @staticmethod
    def url():
        """Generate valid URLs for testing."""
        scheme = random.choice(["http", "https"])
        domain_parts = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(random.randint(1, 3))
        ]
        tld = random.choice(["com", "org", "net", "edu", "gov"])
        path_parts = [
            "".join(
                random.choices(
                    string.ascii_letters + string.digits + "-_", k=random.randint(1, 20)
                )
            )
            for _ in range(random.randint(0, 5))
        ]

        domain = ".".join(domain_parts) + "." + tld
        path = "/" + "/".join(path_parts) if path_parts else ""

        return f"{scheme}://{domain}{path}"

    @staticmethod
    def search_query():
        """Generate realistic search queries."""
        query_type = random.choice(["simple", "phrase", "boolean", "technical"])

        if query_type == "simple":
            # Simple 1-3 word queries
            words = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 12)))
                for _ in range(random.randint(1, 3))
            ]
            return " ".join(words)
        elif query_type == "phrase":
            # Quoted phrase queries
            words = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 10)))
                for _ in range(random.randint(2, 5))
            ]
            return f'"{" ".join(words)}"'
        elif query_type == "boolean":
            # Boolean queries with AND, OR, NOT
            terms = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 10)))
                for _ in range(random.randint(2, 4))
            ]
            operators = random.choices(["AND", "OR", "NOT"], k=len(terms) - 1)

            result = terms[0]
            for i, op in enumerate(operators):
                result += f" {op} {terms[i + 1]}"
            return result
        else:
            # Technical queries with programming terms
            tech_terms = [
                "python",
                "javascript",
                "react",
                "api",
                "database",
                "authentication",
                "docker",
                "kubernetes",
                "microservices",
                "REST",
                "GraphQL",
                "OAuth",
            ]
            selected_terms = random.sample(tech_terms, k=random.randint(1, 3))
            return " ".join(selected_terms)

    @staticmethod
    def embedding_vector(dimension=None):
        """Generate embedding vectors."""
        if dimension is None:
            dimension = random.randint(64, 1536)

        # Generate normalized vector (common for embeddings)
        vector = [random.uniform(-1.0, 1.0) for _ in range(dimension)]

        # Normalize the vector (optional - some embeddings are pre-normalized)
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    @staticmethod
    def document_metadata():
        """Generate document metadata."""
        metadata = {}

        # Always include source
        metadata["source"] = HypothesisStrategies.url()

        # Optional fields
        if random.choice([True, False]):
            metadata["title"] = "".join(
                random.choices(string.ascii_letters + " ", k=random.randint(5, 100))
            )

        if random.choice([True, False]):
            metadata["author"] = fake.name()

        if random.choice([True, False]):
            metadata["published_date"] = fake.date_time_between(
                "-2y", "now"
            ).isoformat()

        if random.choice([True, False]):
            metadata["content_type"] = random.choice(
                ["text/html", "text/plain", "application/pdf", "text/markdown"]
            )

        if random.choice([True, False]):
            metadata["language"] = random.choice(["en", "es", "fr", "de", "zh"])

        if random.choice([True, False]):
            metadata["tags"] = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 15)))
                for _ in range(random.randint(1, 5))
            ]

        return metadata


class TestDataGenerator:
    """High-level test data generator with preset patterns."""

    def __init__(self, seed: int | None = None):
        """Initialize the data generator.

        Args:
            seed: Random seed for reproducible test data
        """
        if seed is not None:
            random.seed(seed)
            fake.seed_instance(seed)

        self.fake = fake

    def generate_document(
        self,
        include_embedding: bool = True,
        embedding_dimension: int = 384,
        content_length_range: tuple = (100, 2000),
    ) -> dict[str, Any]:
        """Generate a realistic document for testing.

        Args:
            include_embedding: Whether to include embedding vector
            embedding_dimension: Dimension of embedding vector
            content_length_range: Min and max content length

        Returns:
            Dictionary representing a document
        """
        doc = {
            "id": str(uuid.uuid4()),
            "content": self._generate_content(content_length_range),
            "metadata": {
                "source": self.fake.url(),
                "title": self.fake.sentence(nb_words=random.randint(3, 8)).rstrip("."),
                "author": self.fake.name(),
                "published_date": self.fake.date_time_between(
                    start_date="-2y", end_date="now"
                ).isoformat(),
                "content_type": random.choice(
                    ["text/html", "text/plain", "application/pdf", "text/markdown"]
                ),
                "language": random.choice(["en", "es", "fr", "de"]),
                "tags": [self.fake.word() for _ in range(random.randint(1, 5))],
            },
            "chunk_index": random.randint(0, 10),
            "chunk_total": random.randint(1, 15),
        }

        if include_embedding:
            doc["embedding"] = self._generate_normalized_vector(embedding_dimension)

        return doc

    def generate_search_result(
        self, query: str, base_score_range: tuple = (0.6, 0.95)
    ) -> dict[str, Any]:
        """Generate a search result for a given query.

        Args:
            query: The search query
            base_score_range: Range for base relevance score

        Returns:
            Dictionary representing a search result
        """
        # Generate base document
        document = self.generate_document(include_embedding=False)

        # Modify content to be somewhat related to query
        query_terms = query.lower().split()[:3]  # Use first 3 terms
        content_parts = [document["content"]]

        # Inject some query terms into content
        for term in query_terms:
            if random.random() < 0.7:  # 70% chance to include each term
                content_parts.append(f" {term}")

        return {
            "id": document["id"],
            "content": "".join(content_parts),
            "score": round(random.uniform(*base_score_range), 4),
            "metadata": document["metadata"],
            "query": query,
            "highlighted_content": self._generate_highlighted_content(
                document["content"], query_terms
            ),
        }

    def generate_search_query(self, query_type: str = "mixed") -> str:
        """Generate a search query of specified type.

        Args:
            query_type: Type of query - "simple", "complex", "technical", or "mixed"

        Returns:
            Generated search query string
        """
        if query_type == "simple":
            return " ".join([self.fake.word() for _ in range(random.randint(1, 3))])

        elif query_type == "complex":
            base_terms = [self.fake.word() for _ in range(random.randint(2, 4))]
            operators = ["AND", "OR", "NOT"]

            query_parts = [base_terms[0]]
            for i in range(1, len(base_terms)):
                op = random.choice(operators)
                query_parts.extend([op, base_terms[i]])

            return " ".join(query_parts)

        elif query_type == "technical":
            tech_terms = [
                "python",
                "javascript",
                "react",
                "api",
                "database",
                "authentication",
                "docker",
                "kubernetes",
                "microservices",
                "REST",
                "GraphQL",
                "OAuth",
                "machine learning",
                "neural network",
                "artificial intelligence",
            ]
            selected = random.sample(tech_terms, random.randint(1, 3))
            return " ".join(selected)

        else:  # mixed
            query_types = ["simple", "complex", "technical"]
            return self.generate_search_query(random.choice(query_types))

    def generate_api_response(
        self,
        success: bool = True,
        include_pagination: bool = False,
        items_count: int | None = None,
    ) -> dict[str, Any]:
        """Generate a mock API response.

        Args:
            success: Whether response represents success or error
            include_pagination: Whether to include pagination metadata
            items_count: Number of items to include (random if None)

        Returns:
            Dictionary representing an API response
        """
        if success:
            response = {
                "status": "success",
                "timestamp": datetime.now(tz=UTC).isoformat() + "Z",
                "request_id": str(uuid.uuid4()),
            }

            if items_count is None:
                items_count = random.randint(0, 20)

            if include_pagination:
                page = random.randint(1, 5)
                per_page = random.randint(10, 50)
                total = items_count + random.randint(0, 100)

                response.update(
                    {
                        "items": [
                            self.generate_document(include_embedding=False)
                            for _ in range(items_count)
                        ],
                        "pagination": {
                            "page": page,
                            "per_page": per_page,
                            "total": total,
                            "total_pages": (total + per_page - 1) // per_page,
                            "has_next": page * per_page < total,
                            "has_prev": page > 1,
                        },
                    }
                )
            else:
                response["data"] = [
                    self.generate_document(include_embedding=False)
                    for _ in range(items_count)
                ]

        else:
            error_types = [
                "validation_error",
                "not_found",
                "internal_error",
                "rate_limit",
            ]
            error_type = random.choice(error_types)

            response = {
                "status": "error",
                "error": error_type,
                "message": self._generate_error_message(error_type),
                "timestamp": datetime.now(tz=UTC).isoformat() + "Z",
                "request_id": str(uuid.uuid4()),
            }

        return response

    def _generate_content(self, length_range: tuple) -> str:
        """Generate realistic content text."""
        paragraphs = []
        total_length = 0
        target_length = random.randint(*length_range)

        while total_length < target_length:
            paragraph = self.fake.paragraph(nb_sentences=random.randint(3, 8))
            paragraphs.append(paragraph)
            total_length += len(paragraph)

        content = " ".join(paragraphs)

        # Truncate to target length if needed
        if len(content) > target_length:
            content = content[: target_length - 3] + "..."

        return content

    def _generate_normalized_vector(self, dimension: int) -> list[float]:
        """Generate a normalized embedding vector."""
        vector = [random.gauss(0, 1) for _ in range(dimension)]
        magnitude = sum(x**2 for x in vector) ** 0.5

        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    def _generate_highlighted_content(self, content: str, terms: list[str]) -> str:
        """Generate content with highlighted search terms."""
        highlighted = content
        for term in terms:
            if term in content.lower():
                # Simple highlighting - replace first occurrence
                highlighted = highlighted.replace(term, f"<mark>{term}</mark>", 1)
        return highlighted

    def _generate_error_message(self, error_type: str) -> str:
        """Generate appropriate error message for error type."""
        messages = {
            "validation_error": "Invalid request parameters provided",
            "not_found": "Requested resource not found",
            "internal_error": "An internal server error occurred",
            "rate_limit": "Rate limit exceeded, please try again later",
        }
        return messages.get(error_type, "An error occurred")


# Convenience functions for quick data generation
def generate_test_documents(count: int = 10, **kwargs) -> list[dict[str, Any]]:
    """Generate a list of test documents.

    Args:
        count: Number of documents to generate
        **kwargs: Additional arguments for TestDataGenerator.generate_document()

    Returns:
        List of generated documents
    """
    generator = TestDataGenerator()
    return [generator.generate_document(**kwargs) for _ in range(count)]


def generate_search_queries(count: int = 10, query_type: str = "mixed") -> list[str]:
    """Generate a list of search queries.

    Args:
        count: Number of queries to generate
        query_type: Type of queries to generate

    Returns:
        List of generated search queries
    """
    generator = TestDataGenerator()
    return [generator.generate_search_query(query_type) for _ in range(count)]
