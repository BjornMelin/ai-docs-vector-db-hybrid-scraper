"""Document chunking primitives tuned for hybrid character/token workflows.

The module blends recursive character and token-aware splitting with semantic
HTML segmentation and lightweight code heuristics to preserve structure during
retrieval-augmented generation (RAG) preprocessing.
"""

# pylint: disable=too-many-lines

import logging
import re
from typing import Any, ClassVar

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Handle both module and script imports
from src.config.models import ChunkingConfig, ChunkingStrategy
from src.models.document_processing import Chunk, CodeBlock, CodeLanguage


class DocumentChunker:
    """Document Chunking Implementation."""

    # Regex patterns for code detection
    CODE_FENCE_PATTERN = re.compile(
        r"(```|~~~)(\w*)\n(.*?)\1", re.DOTALL | re.MULTILINE
    )
    FUNCTION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "python": re.compile(
            r"^(\s*)(async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
            re.MULTILINE,
        ),
        "javascript": re.compile(
            r"^(\s*)(async\s+)?function\s+\w+\s*\([^)]*\)|"
            r"^(\s*)const\s+\w+\s*=\s*(async\s+)?\([^)]*\)\s*=>|"
            r"^(\s*)\w+\s*:\s*(async\s+)?function\s*\([^)]*\)",
            re.MULTILINE,
        ),
        "typescript": re.compile(
            r"^(\s*)(async\s+)?function\s+\w+\s*\([^)]*\)(\s*:\s*[^{]+)?|"
            r"^(\s*)(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?"
            r"\([^)]*\)\s*(?::\s*[^=]+)?\s*=>|"
            r"^(\s*)(public|private|protected)?\s*(async\s+)?\w+\s*\([^)]*\)"
            r"(\s*:\s*[^{]+)?",
            re.MULTILINE,
        ),
    }
    CLASS_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "python": re.compile(r"^(\s*)class\s+\w+(\s*\([^)]*\))?:", re.MULTILINE),
        "javascript": re.compile(r"^(\s*)class\s+\w+(\s+extends\s+\w+)?", re.MULTILINE),
        "typescript": re.compile(
            r"^(\s*)(export\s+)?class\s+\w+(\s+extends\s+\w+)?", re.MULTILINE
        ),
    }
    DOCSTRING_PATTERN = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
    COMMENT_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "python": re.compile(r"^\s*#.*$", re.MULTILINE),
        "javascript": re.compile(r"^\s*//.*$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
        "typescript": re.compile(r"^\s*//.*$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
    }

    # Enhanced boundary patterns
    BOUNDARY_PATTERNS: ClassVar[list[str]] = [
        # Paragraph boundaries
        r"\n\n+",
        # Markdown headers
        r"\n#{1,6}\s+[^\n]+\n",
        # List items
        r"\n\s*[-*+]\s+",
        r"\n\s*\d+\.\s+",
        # Code boundaries
        r"\n```[^\n]*\n",
        r"\n~~~[^\n]*\n",
        # Function/class definitions
        r"\n\s*def\s+",
        r"\n\s*class\s+",
        r"\n\s*function\s+",
        r"\n\s*const\s+\w+\s*=\s*(?:async\s+)?\(",
        # Sentence endings
        r"\.\s+",
        r"[!?]\s+",
        # Documentation sections
        r"\n---+\n",
        r"\n===+\n",
    ]

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize document chunker with configuration.

        Args:
            config: Chunking configuration specifying strategy, sizes, and options.

        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._char_splitter = self._build_character_splitter()
        self._token_splitter = self._build_token_splitter()

    def _build_character_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the character-based splitter respecting configured boundaries."""

        separators = [
            "\n```",
            "\n~~~",
            "\n\n",
            "\n#",
            "\n",
            " ",
            "",
        ]
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=separators,
        )

    def _build_token_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a token-aware splitter using the configured Tiktoken encoder."""

        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.token_model,
                chunk_size=self.config.token_chunk_size,
                chunk_overlap=self.config.token_chunk_overlap,
            )
        except ValueError:  # Unknown encoding -> fall back to characters
            self.logger.warning(
                "Unknown token model '%s'; falling back to character splitter",
                self.config.token_model,
            )
            splitter = self._build_character_splitter()
        return splitter

    def chunk_content(
        self,
        content: str,
        title: str = "",
        url: str = "",
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Main entry point for chunking content with SOTA 2025 strategies."""
        # Detect language if not provided
        if language is None and self.config.detect_language:
            language = self._detect_language(content, url)

        segments = self._segment_input(content)
        chunks: list[Chunk] = []
        running_offset = 0
        for segment in segments:
            segment_chunks = self._semantic_chunking(
                segment, language, base_offset=running_offset
            )
            chunks.extend(segment_chunks)
            running_offset += len(segment)

        # Convert to dict format and add metadata
        return self._format_chunks(chunks, title, url)

    def _segment_input(self, content: str) -> list[str]:
        """Segment raw content based on JSON and HTML heuristics."""

        if self._looks_like_json(content):
            return self._split_json_content(content)
        if self.config.enable_semantic_html_segmentation and self._looks_like_html(
            content
        ):
            return self._extract_html_sections(content)
        return [content]

    @staticmethod
    def _looks_like_html(content: str) -> bool:
        """Rudimentary HTML detection to avoid expensive parsing."""

        snippet = content.strip()[:1000]
        return "<" in snippet and ">" in snippet

    @staticmethod
    def _looks_like_json(content: str) -> bool:
        """Return True when the payload appears to be JSON."""

        stripped = content.lstrip()
        return stripped.startswith("{") or stripped.startswith("[")

    def _split_json_content(self, content: str) -> list[str]:
        """Split JSON payloads into manageable windows based on character count."""

        if len(content) <= self.config.json_max_chars:
            return [content]
        window = self.config.json_max_chars
        return [content[i : i + window] for i in range(0, len(content), window)]

    def _extract_html_sections(self, content: str) -> list[str]:
        """Extract semantic HTML sections using BeautifulSoup when requested."""

        soup = BeautifulSoup(content, "lxml")
        separator = "\n" if self.config.normalize_html_text else "\n\n"
        block_tags = [
            "article",
            "section",
            "main",
            "div",
            "header",
            "footer",
            "aside",
            "nav",
            "pre",
            "code",
            "ul",
            "ol",
            "li",
            "p",
            "table",
        ]
        sections: list[str] = []
        for element in soup.find_all(block_tags):
            text = element.get_text(
                separator=separator, strip=self.config.normalize_html_text
            )
            if text:
                sections.append(text)
        if not sections:
            text = soup.get_text(
                separator=separator, strip=self.config.normalize_html_text
            )
            if text:
                sections.append(text)
        return sections or [content]

    def _detect_language(self, content: str, url: str = "") -> str:
        """Detect programming language from content and URL."""
        # Check file extension in URL
        if (
            url
            and (lang := self._detect_language_from_url(url))
            != CodeLanguage.UNKNOWN.value
        ):
            return lang

        # Check for code fence languages
        if (
            lang := self._detect_language_from_code_fences(content)
        ) != CodeLanguage.UNKNOWN.value:
            return lang

        # Pattern-based detection
        return self._detect_language_from_patterns(content)

    def _detect_language_from_url(self, url: str) -> str:
        """Detect language from URL/file extension."""
        ext_map = {
            ".py": CodeLanguage.PYTHON.value,
            ".js": CodeLanguage.JAVASCRIPT.value,
            ".mjs": CodeLanguage.JAVASCRIPT.value,
            ".ts": CodeLanguage.TYPESCRIPT.value,
            ".tsx": CodeLanguage.TYPESCRIPT.value,
            ".md": CodeLanguage.MARKDOWN.value,
        }
        for ext, lang in ext_map.items():
            if url.endswith(ext):
                return lang
        return CodeLanguage.UNKNOWN.value

    def _detect_language_from_code_fences(self, content: str) -> str:
        """Detect language from code fence declarations."""
        if not (code_fences := self.CODE_FENCE_PATTERN.findall(content)):
            return CodeLanguage.UNKNOWN.value

        # Get most common language
        if not (languages := [fence[1].lower() for fence in code_fences if fence[1]]):
            return CodeLanguage.UNKNOWN.value

        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        detected_lang = max(lang_counts, key=lang_counts.get)  # type: ignore[arg-type]

        lang_map = {
            "python": CodeLanguage.PYTHON.value,
            "py": CodeLanguage.PYTHON.value,
            "javascript": CodeLanguage.JAVASCRIPT.value,
            "js": CodeLanguage.JAVASCRIPT.value,
            "typescript": CodeLanguage.TYPESCRIPT.value,
            "ts": CodeLanguage.TYPESCRIPT.value,
        }
        return lang_map.get(detected_lang, CodeLanguage.UNKNOWN.value)

    def _detect_language_from_patterns(self, content: str) -> str:
        """Detect language from code patterns."""
        if re.search(r"^import\s+\w+|^from\s+\w+\s+import", content, re.MULTILINE):
            return CodeLanguage.PYTHON.value
        if re.search(
            r"^const\s+\w+\s*=|^let\s+\w+\s*=|^var\s+\w+\s*=",
            content,
            re.MULTILINE,
        ):
            return CodeLanguage.JAVASCRIPT.value
        return CodeLanguage.UNKNOWN.value

    def _find_code_blocks(self, content: str) -> list[CodeBlock]:
        """Find all code blocks in content."""
        return [
            CodeBlock(
                language=match.group(2) or "unknown",
                content=match.group(3),
                start_pos=match.start(),
                end_pos=match.end(),
                fence_type=match.group(1),
            )
            for match in self.CODE_FENCE_PATTERN.finditer(content)
        ]

    def _semantic_chunking(
        self, content: str, _language: str | None = None, *, base_offset: int = 0
    ) -> list[Chunk]:
        """Semantic chunking with code awareness but without AST parsing.

        Args:
            content: The text content to chunk.
            language: The detected or specified language (optional).

        Returns:
            List of Chunk objects representing the chunked content.

        """
        chunks = []
        code_blocks = (
            self._find_code_blocks(content) if self.config.preserve_code_blocks else []
        )

        # Sort code blocks by position
        code_blocks.sort(key=lambda b: b.start_pos)

        current_pos = 0
        chunk_start = 0

        while current_pos < len(content):
            # Find the next code block starting at or after current position
            next_code_block = self._get_next_code_block(code_blocks, current_pos)

            if next_code_block and self.config.preserve_code_blocks:
                # Handle any content before the code block
                if chunk_start < next_code_block.start_pos:
                    pre_content = content[
                        chunk_start : next_code_block.start_pos
                    ].strip()
                    if pre_content:
                        chunks.extend(
                            self._chunk_text_content(
                                pre_content, base_offset + chunk_start
                            )
                        )

                # Handle the code block
                self._handle_code_block_as_chunk(
                    content, chunks, next_code_block, base_offset
                )
                current_pos = next_code_block.end_pos
                chunk_start = current_pos
            else:
                # Handle remaining content as regular text
                if remaining_content := content[chunk_start:].strip():
                    chunks.extend(
                        self._chunk_text_content(
                            remaining_content, base_offset + chunk_start
                        )
                    )
                break
        # Update total chunks count
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.total_chunks = len(chunks)
            chunk.char_count = len(chunk.content)
            chunk.token_estimate = chunk.char_count // 4
        return chunks

    def _get_current_code_block(self, code_blocks, current_pos):
        """Return the code block at the current position, if any.

        Args:
            code_blocks: List of CodeBlock objects.
            current_pos: Current character position in content.

        Returns:
            The CodeBlock at current_pos, or None if not in a code block.

        """
        for block in code_blocks:
            if block.start_pos <= current_pos < block.end_pos:
                return block
        return None

    def _get_next_code_block(self, code_blocks, current_pos):
        """Return the next code block starting at or after current position.

        Args:
            code_blocks: List of CodeBlock objects.
            current_pos: Current character position in content.

        Returns:
            The next CodeBlock starting at or after current_pos, or None.

        """
        for block in code_blocks:
            if block.start_pos >= current_pos:
                return block
        return None

    def _handle_code_block_as_chunk(
        self, content: str, chunks: list[Chunk], code_block: CodeBlock, base_offset: int
    ) -> None:
        """Handle a code block as a single chunk.

        Args:
            content: The full text content.
            chunks: List to append new chunks to.
            code_block: The CodeBlock being processed.

        """
        start = base_offset + code_block.start_pos
        end = base_offset + code_block.end_pos
        block_content = content[code_block.start_pos : code_block.end_pos]

        if (code_block.end_pos - code_block.start_pos) <= self.config.max_chunk_size:
            chunks.append(
                Chunk(
                    content=block_content,
                    start_pos=start,
                    end_pos=end,
                    chunk_index=len(chunks),
                    chunk_type="code",
                    language=code_block.language,
                    has_code=True,
                )
            )
        else:
            # Code block too large, split it preserving boundaries
            chunks.extend(
                self._chunk_large_code_block(
                    block_content,
                    start,
                    code_block.language,
                )
            )

    def _find_semantic_boundary(self, content: str, start: int, target_end: int) -> int:
        """Find a semantic boundary considering code structures."""
        search_window = 200
        search_start = max(target_end - search_window, start)

        best_boundary = target_end
        best_score = 0

        # Try each boundary pattern
        for pattern in self.BOUNDARY_PATTERNS:
            if matches := list(re.finditer(pattern, content[search_start:target_end])):
                # Get the last match (closest to target)
                match = matches[-1]
                boundary_pos = search_start + match.end()

                # Score based on proximity to target and boundary type
                distance_score = 1 - (abs(target_end - boundary_pos) / search_window)

                # Higher scores for stronger boundaries
                if "def " in pattern or "class " in pattern or "function" in pattern:
                    type_score = 1.0
                elif "\\n\\n" in pattern or "#{" in pattern:
                    type_score = 0.8
                elif ". " in pattern or "! " in pattern:
                    type_score = 0.6
                else:
                    type_score = 0.4

                if (score := distance_score * type_score) > best_score:
                    best_score = score
                    best_boundary = boundary_pos

        return best_boundary

    def _chunk_text_content(self, content: str, global_start: int) -> list[Chunk]:
        """Chunk text content using the configured splitter strategy."""

        if not content:
            return []

        splitter = (
            self._token_splitter
            if self.config.strategy is ChunkingStrategy.ENHANCED
            else self._char_splitter
        )
        splits = splitter.split_text(content)
        chunks: list[Chunk] = []
        cursor = 0

        for split in splits:
            if not split.strip():
                cursor += len(split)
                continue
            relative_start = content.find(split, cursor)
            if relative_start == -1:
                relative_start = cursor
            relative_end = relative_start + len(split)
            cursor = relative_end
            chunks.append(
                Chunk(
                    content=split,
                    start_pos=global_start + relative_start,
                    end_pos=global_start + relative_end,
                    chunk_index=0,
                    chunk_type="text",
                )
            )

        return chunks

    def _chunk_large_code_block(  # pylint: disable=too-many-locals
        self, code_content: str, global_start: int, language: str
    ) -> list[Chunk]:
        """Chunk large code blocks while preserving structure."""
        chunks = []

        # Try to split on function boundaries if we know the language
        if language in self.FUNCTION_PATTERNS:
            function_pattern = self.FUNCTION_PATTERNS[language]

            if functions := list(function_pattern.finditer(code_content)):
                # Chunk by functions
                for i, match in enumerate(functions):
                    func_start = match.start()

                    # Find end of function (next function or end of block)
                    if i + 1 < len(functions):
                        func_end = functions[i + 1].start()
                    else:
                        func_end = len(code_content)

                    if func_content := code_content[func_start:func_end].strip():
                        chunks.append(
                            Chunk(
                                content=func_content,
                                start_pos=global_start + func_start,
                                end_pos=global_start + func_end,
                                chunk_index=0,  # Will be updated
                                chunk_type="code",
                                language=language,
                                has_code=True,
                                metadata={"is_function": True},
                            )
                        )

                # Handle any code before first function
                if (
                    functions
                    and functions[0].start() > 0
                    and (pre_func := code_content[: functions[0].start()].strip())
                ):
                    chunks.insert(
                        0,
                        Chunk(
                            content=pre_func,
                            start_pos=global_start,
                            end_pos=global_start + functions[0].start(),
                            chunk_index=0,
                            chunk_type="code",
                            language=language,
                            has_code=True,
                        ),
                    )

                return chunks

        # Fallback: chunk by lines with smart boundaries
        lines = code_content.split("\n")
        current_chunk_lines = []
        current_size = 0
        chunk_start_pos = global_start

        for _i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed limit
            if (
                current_size + line_size > self.config.chunk_size
                and current_chunk_lines
            ):
                # Create chunk
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        start_pos=chunk_start_pos,
                        end_pos=chunk_start_pos + len(chunk_content),
                        chunk_index=0,
                        chunk_type="code",
                        language=language,
                        has_code=True,
                    )
                )

                # Start new chunk with overlap (include last few lines)
                overlap_lines = max(1, len(current_chunk_lines) // 5)  # 20% overlap
                current_chunk_lines = current_chunk_lines[-overlap_lines:]
                current_size = sum(len(line) + 1 for line in current_chunk_lines)
                chunk_start_pos += len(chunk_content) - current_size

            current_chunk_lines.append(line)
            current_size += line_size

        # Add remaining lines
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                Chunk(
                    content=chunk_content,
                    start_pos=chunk_start_pos,
                    end_pos=global_start + len(code_content),
                    chunk_index=0,
                    chunk_type="code",
                    language=language,
                    has_code=True,
                )
            )

        return chunks

    def _format_chunks(
        self, chunks: list[Chunk], title: str, url: str
    ) -> list[dict[str, Any]]:
        """Format chunks into dictionary format expected by the system."""
        formatted = []

        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk.content,
                "title": (f"{title} (Part {i + 1})" if i > 0 and title else title),
                "url": url,
                "chunk_index": i,
                "total_chunks": chunk.total_chunks,
                "char_count": chunk.char_count,
                "token_estimate": chunk.token_estimate,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "chunk_type": chunk.chunk_type,
                "has_code": chunk.has_code,
            }

            # Add optional fields
            if chunk.language:
                chunk_dict["language"] = chunk.language
            if chunk.metadata:
                chunk_dict["metadata"] = chunk.metadata

            formatted.append(chunk_dict)

        return formatted
