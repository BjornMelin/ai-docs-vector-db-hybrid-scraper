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
            return random.choice(names)  # noqa: S311

        @staticmethod
        def url():
            domains = ["example.com", "test.org", "demo.net"]
            return f"https://{random.choice(domains)}/page{random.randint(1, 100)}"  # noqa: S311

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
            return " ".join(random.choices(words, k=nb_words)).capitalize() + "."  # noqa: S311

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
            return random.choice(words)  # noqa: S311

        @staticmethod
        def date_time_between(_start_date, _end_date):
            # Simple date generation - uses fixed range for now
            start = datetime.now(tz=UTC) - timedelta(days=365)
            end = datetime.now(tz=UTC)
            return start + timedelta(
                seconds=random.randint(0, int((end - start)._total_seconds()))  # noqa: S311
            )

        @staticmethod
        def image_url():
            return (
                f"https://via.placeholder.com/{random.randint(100, 800)}x"  # noqa: S311
                f"{random.randint(100, 600)}"  # noqa: S311
            )

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
        id_type = random.choice(id_types)  # noqa: S311

        if id_type == "uuid":
            return str(uuid.uuid4())
        if id_type == "numeric":
            return str(random.randint(1, 999999))  # noqa: S311
        # Custom alphanumeric ID
        length = random.randint(8, 32)  # noqa: S311
        chars = string.ascii_letters + string.digits + "_-"
        return "".join(random.choices(chars, k=length))  # noqa: S311

    @staticmethod
    def url():
        """Generate valid URLs for testing."""
        scheme = random.choice(["http", "https"])  # noqa: S311
        domain_parts = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))  # noqa: S311
            for _ in range(random.randint(1, 3))  # noqa: S311
        ]
        tld = random.choice(["com", "org", "net", "edu", "gov"])  # noqa: S311
        path_parts = [
            "".join(
                random.choices(  # noqa: S311
                    string.ascii_letters + string.digits + "-_",
                    k=random.randint(1, 20),  # noqa: S311
                )
            )
            for _ in range(random.randint(0, 5))  # noqa: S311
        ]

        domain = ".".join(domain_parts) + "." + tld
        path = "/" + "/".join(path_parts) if path_parts else ""

        return f"{scheme}://{domain}{path}"

    @staticmethod
    def search_query():
        """Generate realistic search queries."""
        query_type = random.choice(["simple", "phrase", "boolean", "technical"])  # noqa: S311

        if query_type == "simple":
            # Simple 1-3 word queries
            words = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 12)))  # noqa: S311
                for _ in range(random.randint(1, 3))  # noqa: S311
            ]
            return " ".join(words)
        if query_type == "phrase":
            # Quoted phrase queries
            words = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 10)))  # noqa: S311
                for _ in range(random.randint(2, 5))  # noqa: S311
            ]
            return f'"{" ".join(words)}"'
        if query_type == "boolean":
            # Boolean queries with AND, OR, NOT
            terms = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 10)))  # noqa: S311
                for _ in range(random.randint(2, 4))  # noqa: S311
            ]
            operators = random.choices(["AND", "OR", "NOT"], k=len(terms) - 1)  # noqa: S311

            result = terms[0]
            for i, op in enumerate(operators):
                result += f" {op} {terms[i + 1]}"
            return result
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
        selected_terms = random.sample(tech_terms, k=random.randint(1, 3))  # noqa: S311
        return " ".join(selected_terms)

    @staticmethod
    def embedding_vector(dimension=None):
        """Generate embedding vectors."""
        if dimension is None:
            dimension = random.randint(64, 1536)  # noqa: S311

        # Generate normalized vector (common for embeddings)
        vector = [random.uniform(-1.0, 1.0) for _ in range(dimension)]  # noqa: S311

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
        if random.choice([True, False]):  # noqa: S311
            metadata["title"] = "".join(
                random.choices(string.ascii_letters + " ", k=random.randint(5, 100))  # noqa: S311
            )

        if random.choice([True, False]):  # noqa: S311
            metadata["author"] = fake.name()

        if random.choice([True, False]):  # noqa: S311
            metadata["published_date"] = fake.date_time_between(
                "-2y", "now"
            ).isoformat()

        if random.choice([True, False]):  # noqa: S311
            metadata["content_type"] = random.choice(  # noqa: S311
                ["text/html", "text/plain", "application/pdf", "text/markdown"]
            )

        if random.choice([True, False]):  # noqa: S311
            metadata["language"] = random.choice(["en", "es", "fr", "de", "zh"])  # noqa: S311

        if random.choice([True, False]):  # noqa: S311
            metadata["tags"] = [
                "".join(random.choices(string.ascii_letters, k=random.randint(3, 15)))  # noqa: S311
                for _ in range(random.randint(1, 5))  # noqa: S311
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
            random.seed(seed)  # noqa: S311
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
                "title": self.fake.sentence(nb_words=random.randint(3, 8)).rstrip("."),  # noqa: S311
                "author": self.fake.name(),
                "published_date": self.fake.date_time_between(
                    start_date="-2y", end_date="now"
                ).isoformat(),
                "content_type": random.choice(  # noqa: S311
                    ["text/html", "text/plain", "application/pdf", "text/markdown"]
                ),
                "language": random.choice(["en", "es", "fr", "de"]),  # noqa: S311
                "tags": [self.fake.word() for _ in range(random.randint(1, 5))],  # noqa: S311
            },
            "chunk_index": random.randint(0, 10),  # noqa: S311
            "chunk__total": random.randint(1, 15),  # noqa: S311
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
        content_parts.extend(
            f" {term}"
            for term in query_terms
            if random.random() < 0.7  # noqa: S311
        )

        return {
            "id": document["id"],
            "content": "".join(content_parts),
            "score": round(random.uniform(*base_score_range), 4),  # noqa: S311
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
            return " ".join([self.fake.word() for _ in range(random.randint(1, 3))])  # noqa: S311

        if query_type == "complex":
            base_terms = [self.fake.word() for _ in range(random.randint(2, 4))]  # noqa: S311
            operators = ["AND", "OR", "NOT"]

            query_parts = [base_terms[0]]
            for i in range(1, len(base_terms)):
                op = random.choice(operators)  # noqa: S311
                query_parts.extend([op, base_terms[i]])

            return " ".join(query_parts)

        if query_type == "technical":
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
            selected = random.sample(tech_terms, random.randint(1, 3))  # noqa: S311
            return " ".join(selected)

        # mixed
        query_types = ["simple", "complex", "technical"]
        return self.generate_search_query(random.choice(query_types))  # noqa: S311

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
                items_count = random.randint(0, 20)  # noqa: S311

            if include_pagination:
                page = random.randint(1, 5)  # noqa: S311
                per_page = random.randint(10, 50)  # noqa: S311
                _total = items_count + random.randint(0, 100)  # noqa: S311

                response.update(
                    {
                        "items": [
                            self.generate_document(include_embedding=False)
                            for _ in range(items_count)
                        ],
                        "pagination": {
                            "page": page,
                            "per_page": per_page,
                            "_total": _total,
                            "_total_pages": (_total + per_page - 1) // per_page,
                            "has_next": page * per_page < _total,
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
            error_type = random.choice(error_types)  # noqa: S311

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
        _total_length = 0
        target_length = random.randint(*length_range)  # noqa: S311

        while _total_length < target_length:
            paragraph = self.fake.paragraph(nb_sentences=random.randint(3, 8))  # noqa: S311
            paragraphs.append(paragraph)
            _total_length += len(paragraph)

        content = " ".join(paragraphs)

        # Truncate to target length if needed
        if len(content) > target_length:
            content = content[: target_length - 3] + "..."

        return content

    def _generate_normalized_vector(self, dimension: int) -> list[float]:
        """Generate a normalized embedding vector."""
        vector = [random.gauss(0, 1) for _ in range(dimension)]  # noqa: S311
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
def generate_test_documents(count: int = 10, **_kwargs) -> list[dict[str, Any]]:
    """Generate a list of test documents.

    Args:
        count: Number of documents to generate
        **_kwargs: Additional arguments for TestDataGenerator.generate_document()

    Returns:
        List of generated documents
    """
    generator = TestDataGenerator()
    return [generator.generate_document(**_kwargs) for _ in range(count)]


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
