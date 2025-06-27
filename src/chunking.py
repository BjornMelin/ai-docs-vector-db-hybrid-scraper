"""chunking.py - Document Chunking with Code Awareness.

Implements research-backed chunking strategies for optimal RAG performance.
Supports semantic boundary detection, code-aware chunking, and Tree-sitter AST parsing.
"""

import logging
import re
from typing import Any, ClassVar

# Handle both module and script imports
from src.config import ChunkingConfig, ChunkingStrategy
from src.models.document_processing import Chunk, CodeBlock, CodeLanguage


try:
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True

    # Try to import language parsers using modern syntax
    try:
        import tree_sitter_python as tspython

        PYTHON_AVAILABLE = True
    except ImportError:
        PYTHON_AVAILABLE = False
        tspython = None  # type: ignore

    try:
        import tree_sitter_javascript as tsjavascript

        JAVASCRIPT_AVAILABLE = True
    except ImportError:
        JAVASCRIPT_AVAILABLE = False
        tsjavascript = None  # type: ignore

    try:
        import tree_sitter_typescript as tstypescript

        TYPESCRIPT_AVAILABLE = True
    except ImportError:
        TYPESCRIPT_AVAILABLE = False
        tstypescript = None  # type: ignore

except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = None  # type: ignore
    Node = None  # type: ignore
    PYTHON_AVAILABLE = False
    JAVASCRIPT_AVAILABLE = False
    TYPESCRIPT_AVAILABLE = False
    tspython = None  # type: ignore
    tsjavascript = None  # type: ignore
    tstypescript = None  # type: ignore


class DocumentChunker:
    """Document Chunking Implementation"""

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
        self.config = config
        self.parsers: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        """Initialize Tree-sitter parsers for all supported languages.

        This method attempts to load parsers for each language specified in
        self.config.supported_languages. If a parser fails to load, it logs
        a warning and continues with other languages. The successfully loaded
        parsers are stored in self.parsers dictionary, keyed by language name.
        """
        if not TREE_SITTER_AVAILABLE or not self.config.enable_ast_chunking:
            return

        # Map language names to their parser modules and availability flags
        language_parsers = {
            "python": (tspython, PYTHON_AVAILABLE),
            "javascript": (tsjavascript, JAVASCRIPT_AVAILABLE),
            "typescript": (tstypescript, TYPESCRIPT_AVAILABLE),
        }

        for lang in self.config.supported_languages:
            if lang not in language_parsers:
                self.logger.warning(
                    f"Language '{lang}' is not supported for AST parsing. "
                    f"Supported languages: {list(language_parsers.keys())}"
                )
                continue

            parser_module, is_available = language_parsers[lang]

            if not is_available or parser_module is None:
                self.logger.warning(
                    f"Parser for '{lang}' is not available. "
                    f"Install with: pip install tree-sitter-{lang}"
                )
                continue

            try:
                # Initialize language using modern Tree-sitter pattern
                if lang == "typescript":
                    # TypeScript has specific function name
                    language = Language(parser_module.language_typescript())
                elif lang == "javascript":
                    # JavaScript standard function name
                    language = Language(parser_module.language())
                elif lang == "python":
                    # Python standard function name
                    language = Language(parser_module.language())
                else:
                    # Fallback to standard function name
                    language = Language(parser_module.language())

                # Create parser with the language
                parser = Parser(language)
                self.parsers[lang] = parser
                self.logger.debug(f"Successfully loaded parser for '{lang}'")

            except Exception:
                self.logger.warning(
                    f"Failed to initialize parser for '{lang}': {e}. "
                    f"Will fall back to semantic chunking for this language."
                )

    def chunk_content(
        self,
        content: str,
        title: str = "",
        url: str = "",
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Main entry point for chunking content with SOTA 2025 strategies"""
        # Detect language if not provided
        if language is None and self.config.detect_language:
            language = self._detect_language(content, url)

        # Choose chunking strategy
        if (
            self.config.strategy == ChunkingStrategy.AST_AWARE
            and TREE_SITTER_AVAILABLE
            and language in self.parsers
        ):
            chunks = self._ast_based_chunking(content, language)
        elif self.config.strategy == ChunkingStrategy.ENHANCED:
            chunks = self._semantic_chunking(content, language)
        else:
            chunks = self._basic_chunking(content)

        # Convert to dict format and add metadata
        return self._format_chunks(chunks, title, url)

    def _detect_language(self, content: str, url: str = "") -> str:
        """Detect programming language from content and URL"""
        # Check file extension in URL
        if url:
            lang = self._detect_language_from_url(url)
            if lang != CodeLanguage.UNKNOWN.value:
                return lang

        # Check for code fence languages
        lang = self._detect_language_from_code_fences(content)
        if lang != CodeLanguage.UNKNOWN.value:
            return lang

        # Pattern-based detection
        return self._detect_language_from_patterns(content)

    def _detect_language_from_url(self, url: str) -> str:
        """Detect language from URL/file extension"""
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
        """Detect language from code fence declarations"""
        code_fences = self.CODE_FENCE_PATTERN.findall(content)
        if not code_fences:
            return CodeLanguage.UNKNOWN.value

        # Get most common language
        languages = [fence[1].lower() for fence in code_fences if fence[1]]
        if not languages:
            return CodeLanguage.UNKNOWN.value

        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        detected_lang = max(lang_counts, key=lang_counts.get)  # type: ignore

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
        """Detect language from code patterns"""
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
        """Find all code blocks in content"""
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
        self, content: str, _language: str | None = None
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
                                pre_content, chunk_start, next_code_block.start_pos
                            )
                        )

                # Handle the code block
                self._handle_code_block_as_chunk(content, chunks, next_code_block)
                current_pos = next_code_block.end_pos
                chunk_start = current_pos
            else:
                # Handle remaining content as regular text
                remaining_content = content[chunk_start:].strip()
                if remaining_content:
                    chunks.extend(
                        self._chunk_text_content(
                            remaining_content, chunk_start, len(content)
                        )
                    )
                break
        # Update total chunks count
        for chunk in chunks:
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

    def _handle_code_block_as_chunk(self, content, chunks, code_block):
        """Handle a code block as a single chunk.

        Args:
            content: The full text content.
            chunks: List to append new chunks to.
            code_block: The CodeBlock being processed.
        """
        start = code_block.start_pos
        end = code_block.end_pos
        block_size = end - start

        if block_size <= self.config.max_function_chunk_size:
            chunks.append(
                Chunk(
                    content=content[start:end],
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
                    content[start:end],
                    start,
                    code_block.language,
                )
            )

    def _find_semantic_boundary(self, content: str, start: int, target_end: int) -> int:
        """Find a semantic boundary considering code structures"""
        search_window = 200
        search_start = max(target_end - search_window, start)

        best_boundary = target_end
        best_score = 0

        # Try each boundary pattern
        for pattern in self.BOUNDARY_PATTERNS:
            matches = list(re.finditer(pattern, content[search_start:target_end]))
            if matches:
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

                score = distance_score * type_score

                if score > best_score:
                    best_score = score
                    best_boundary = boundary_pos

        return best_boundary

    def _chunk_text_content(
        self, content: str, global_start: int, _global_end: int
    ) -> list[Chunk]:
        """Chunk text content with overlap"""
        chunks = []
        local_pos = 0

        while local_pos < len(content):
            chunk_end = min(local_pos + self.config.chunk_size, len(content))

            # Find boundary
            if chunk_end < len(content):
                chunk_end = self._find_text_boundary(content, local_pos, chunk_end)

            chunk_content = content[local_pos:chunk_end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        start_pos=global_start + local_pos,
                        end_pos=global_start + chunk_end,
                        chunk_index=0,  # Will be updated
                        chunk_type="text",
                    )
                )

            # Move with overlap
            if chunk_end < len(content):
                local_pos = chunk_end - self.config.chunk_overlap
            else:
                local_pos = len(content)

        return chunks

    def _find_text_boundary(self, content: str, start: int, target_end: int) -> int:
        """Find a good text boundary (sentence, paragraph, etc.)"""
        search_window = min(200, target_end - start)
        search_start = max(target_end - search_window, start)

        # Look for boundaries in order of preference
        boundaries = [
            (r"\.\s+", 1.0),  # Sentence end
            (r"[!?]\s+", 0.9),  # Exclamation or question
            (r"\n\n+", 0.8),  # Paragraph
            (r"\n", 0.6),  # Line break
            (r",\s+", 0.4),  # Comma
            (r";\s+", 0.5),  # Semicolon
        ]

        best_pos = target_end
        for pattern, _score in boundaries:
            matches = list(re.finditer(pattern, content[search_start:target_end]))
            if matches:
                # Use the last match
                last_match = matches[-1]
                boundary_pos = search_start + last_match.end()
                if boundary_pos > start + 100:  # Ensure minimum chunk size
                    best_pos = boundary_pos
                    break

        return best_pos

    def _chunk_large_code_block(
        self, code_content: str, global_start: int, language: str
    ) -> list[Chunk]:
        """Chunk large code blocks while preserving structure"""
        chunks = []

        # Try to split on function boundaries if we know the language
        if language in self.FUNCTION_PATTERNS:
            function_pattern = self.FUNCTION_PATTERNS[language]
            functions = list(function_pattern.finditer(code_content))

            if functions:
                # Chunk by functions
                for i, match in enumerate(functions):
                    func_start = match.start()

                    # Find end of function (next function or end of block)
                    if i + 1 < len(functions):
                        func_end = functions[i + 1].start()
                    else:
                        func_end = len(code_content)

                    func_content = code_content[func_start:func_end].strip()
                    if func_content:
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
                if functions and functions[0].start() > 0:
                    pre_func = code_content[: functions[0].start()].strip()
                    if pre_func:
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

    def _ast_based_chunking(self, content: str, language: str) -> list[Chunk]:
        """AST-based chunking using Tree-sitter for superior code understanding.

        This method uses Tree-sitter parsers to create chunks based on the Abstract
        Syntax Tree (AST) structure of the code. It extracts semantic units like
        functions and classes as individual chunks.

        Overlap Strategy:
        -----------------
        When AST chunking is active, overlap is applied in the following scenarios:

        1. **Text content between code units**: Non-code content uses standard
           character-based overlap as defined by self.config.chunk_overlap.

        2. **Large code units**: When a function or class exceeds max_function_chunk_size,
           it is split with overlap. The overlap includes:
           - For line-based splitting: ~20% of lines from the previous chunk
           - For AST-based splitting: Context like function signatures or class
             names are preserved across splits when possible

        3. **Semantic context preservation**: When splitting large units, the chunker
           attempts to maintain semantic context by including parent node information
           (e.g., class name when splitting methods).

        Args:
            content: The source code to chunk
            language: The programming language of the content

        Returns:
            List of Chunk objects with AST-based boundaries
        """
        if not TREE_SITTER_AVAILABLE or language not in self.parsers:
            return self._semantic_chunking(content, language)

        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node

            # Extract all function and class nodes
            code_units = self._extract_code_units(root_node, content, language)

            if not code_units:
                # No significant code structures found, use semantic chunking
                return self._semantic_chunking(content, language)

            # Sort by position in file
            code_units.sort(key=lambda unit: unit["start_pos"])

            chunks = []
            last_end = 0

            for unit in code_units:
                # Handle any content before this code unit
                if unit["start_pos"] > last_end:
                    pre_content = content[last_end : unit["start_pos"]].strip()
                    if pre_content:
                        # Chunk the non-code content
                        pre_chunks = self._chunk_text_content(
                            pre_content, last_end, unit["start_pos"]
                        )
                        chunks.extend(pre_chunks)

                # Add the code unit as a chunk
                unit_content = content[unit["start_pos"] : unit["end_pos"]]

                # Check if the unit is too large
                if len(unit_content) <= self.config.max_function_chunk_size:
                    chunks.append(
                        Chunk(
                            content=unit_content,
                            start_pos=unit["start_pos"],
                            end_pos=unit["end_pos"],
                            chunk_index=len(chunks),
                            chunk_type="code",
                            language=language,
                            has_code=True,
                            metadata={
                                "node_type": unit["type"],
                                "name": unit.get("name", ""),
                            },
                        )
                    )
                else:
                    # Split large code units intelligently
                    sub_chunks = self._split_large_code_unit(
                        unit_content,
                        unit["start_pos"],
                        unit["type"],
                        language,
                    )
                    chunks.extend(sub_chunks)

                last_end = unit["end_pos"]

            # Handle any remaining content
            if last_end < len(content):
                remaining = content[last_end:].strip()
                if remaining:
                    remaining_chunks = self._chunk_text_content(
                        remaining, last_end, len(content)
                    )
                    chunks.extend(remaining_chunks)

            # Update chunk indices and metadata
            for i, chunk in enumerate(chunks):
                chunk.chunk_index = i
                chunk.total_chunks = len(chunks)
                chunk.char_count = len(chunk.content)
                chunk.token_estimate = chunk.char_count // 4

            return chunks

        except Exception:
            # Fall back to semantic chunking on error
            return self._semantic_chunking(content, language)

    def _extract_code_units(
        self, node: Any, content: str, language: str
    ) -> list[dict[str, Any]]:
        """Extract function and class definitions from AST.

        Args:
            node: The root AST node.
            content: The full text content.
            language: The detected or specified language.

        Returns:
            List of dicts with code unit metadata (type, name, start_pos, end_pos).
        """
        code_units = []

        def traverse(node: Any) -> None:
            # Python-specific node types
            if language == "python":
                self._traverse_python(node, content, code_units)
            elif language in ["javascript", "typescript"]:
                self._traverse_js_ts(node, content, code_units, language)
            for child in node.children:
                traverse(child)

        traverse(node)
        return code_units

    def _traverse_python(
        self, node: Any, content: str, code_units: list[dict[str, Any]]
    ):
        """Helper for traversing Python AST nodes and extracting code units.

        Args:
            node: The AST node.
            content: The full text content.
            code_units: List to append code unit dicts to.
        """
        if node.type in ["function_definition", "async_function_definition"]:
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            code_units.append(
                {
                    "type": "function",
                    "name": (
                        content[name_node.start_byte : name_node.end_byte]
                        if name_node
                        else ""
                    ),
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            )
        elif node.type == "class_definition":
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            code_units.append(
                {
                    "type": "class",
                    "name": (
                        content[name_node.start_byte : name_node.end_byte]
                        if name_node
                        else ""
                    ),
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            )

    def _traverse_js_ts(
        self, node: Any, content: str, code_units: list[dict[str, Any]], _language: str
    ):
        """Helper for traversing JavaScript/TypeScript AST nodes and extracting code units.

        Args:
            node: The AST node.
            content: The full text content.
            code_units: List to append code unit dicts to.
            language: The detected or specified language.
        """
        if node.type in [
            "function_declaration",
            "function_expression",
            "arrow_function",
        ]:
            name = ""
            if node.type == "function_declaration":
                for child in node.children:
                    if child.type == "identifier":
                        name = content[child.start_byte : child.end_byte]
                        break
            code_units.append(
                {
                    "type": "function",
                    "name": name,
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            )
        elif node.type == "class_declaration":
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = content[child.start_byte : child.end_byte]
                    break
            code_units.append(
                {
                    "type": "class",
                    "name": name,
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            )

    def _split_large_code_unit(
        self, content: str, global_start: int, unit_type: str, language: str
    ) -> list[Chunk]:
        """Split large code units using AST-specific logic.

        This method intelligently splits large functions or classes based on their
        AST structure rather than simple line-based splitting. It attempts to:

        1. For classes: Split by method boundaries
        2. For functions: Split by logical blocks (loops, conditionals)
        3. Preserve semantic context across splits

        Args:
            content: The code unit content to split
            global_start: Starting position in the original document
            unit_type: Type of code unit ('class' or 'function')
            language: Programming language

        Returns:
            List of Chunk objects representing the split code unit
        """
        chunks = []

        # If we have a parser for this language, try AST-based splitting
        if language in self.parsers:
            try:
                parser = self.parsers[language]
                tree = parser.parse(bytes(content, "utf8"))
                root_node = tree.root_node

                if unit_type == "class":
                    # For classes, try to split by methods
                    method_nodes = self._extract_class_methods(
                        root_node, content, language
                    )

                    if method_nodes:
                        # Add class header as first chunk if it exists
                        first_method_start = method_nodes[0]["start_pos"]
                        if first_method_start > 0:
                            class_header = content[:first_method_start].strip()
                            if class_header:
                                chunks.append(
                                    Chunk(
                                        content=class_header,
                                        start_pos=global_start,
                                        end_pos=global_start + first_method_start,
                                        chunk_index=0,
                                        chunk_type="code",
                                        language=language,
                                        has_code=True,
                                        metadata={
                                            "node_type": "class_header",
                                            "parent_type": unit_type,
                                        },
                                    )
                                )

                        # Add each method as a chunk, with class context
                        for method in method_nodes:
                            method_content = content[
                                method["start_pos"] : method["end_pos"]
                            ]

                            # If method is still too large, recursively split it
                            if (
                                len(method_content)
                                > self.config.max_function_chunk_size
                            ):
                                sub_chunks = self._split_large_code_unit(
                                    method_content,
                                    global_start + method["start_pos"],
                                    "function",
                                    language,
                                )
                                # Add class context to sub-chunks
                                for sub_chunk in sub_chunks:
                                    if sub_chunk.metadata is None:
                                        sub_chunk.metadata = {}
                                    sub_chunk.metadata["parent_class"] = True
                                chunks.extend(sub_chunks)
                            else:
                                chunks.append(
                                    Chunk(
                                        content=method_content,
                                        start_pos=global_start + method["start_pos"],
                                        end_pos=global_start + method["end_pos"],
                                        chunk_index=0,
                                        chunk_type="code",
                                        language=language,
                                        has_code=True,
                                        metadata={
                                            "node_type": "method",
                                            "method_name": method.get("name", ""),
                                            "parent_type": unit_type,
                                        },
                                    )
                                )

                        return chunks

                elif unit_type == "function":
                    # For functions, try to split by logical blocks
                    blocks = self._extract_function_blocks(root_node, content, language)

                    if len(blocks) > 1:
                        # Add function signature as context for all chunks
                        func_signature = self._extract_function_signature(
                            root_node, content, language
                        )

                        for i, block in enumerate(blocks):
                            block_content = content[
                                block["start_pos"] : block["end_pos"]
                            ]

                            # Include function signature as comment for context (except first block)
                            if i > 0 and func_signature:
                                block_content = (
                                    f"# Function: {func_signature}\n{block_content}"
                                )

                            chunks.append(
                                Chunk(
                                    content=block_content,
                                    start_pos=global_start + block["start_pos"],
                                    end_pos=global_start + block["end_pos"],
                                    chunk_index=0,
                                    chunk_type="code",
                                    language=language,
                                    has_code=True,
                                    metadata={
                                        "node_type": "function_block",
                                        "block_type": block.get("type", ""),
                                        "parent_type": unit_type,
                                        "block_index": i,
                                    },
                                )
                            )

                        if chunks:
                            return chunks

            except Exception:
                self.logger.debug(
                    f"AST-based splitting failed: {e}, falling back to line-based"
                )

        # Fallback to line-based splitting with semantic overlap
        return self._chunk_large_code_block(content, global_start, language)

    def _extract_class_methods(
        self, node: Any, content: str, language: str
    ) -> list[dict[str, Any]]:
        """Extract method definitions from a class AST node.

        Args:
            node: The class AST node
            content: The source code content
            language: Programming language

        Returns:
            List of dicts containing method info (name, start_pos, end_pos)
        """
        methods = []

        def traverse(node: Any) -> None:
            if language == "python":
                if node.type in ["function_definition", "async_function_definition"]:
                    # Check if this is a method (has proper indentation for class)
                    name_node = None
                    for child in node.children:
                        if child.type == "identifier":
                            name_node = child
                            break

                    methods.append(
                        {
                            "name": content[name_node.start_byte : name_node.end_byte]
                            if name_node
                            else "",
                            "start_pos": node.start_byte,
                            "end_pos": node.end_byte,
                            "type": "method",
                        }
                    )
            elif (
                language in ["javascript", "typescript"]
                and node.type == "method_definition"
            ):
                # Methods in JS/TS classes
                methods.append(
                    {
                        "name": self._get_js_method_name(node, content),
                        "start_pos": node.start_byte,
                        "end_pos": node.end_byte,
                        "type": "method",
                    }
                )

            for child in node.children:
                traverse(child)

        traverse(node)
        return methods

    def _extract_function_blocks(
        self, node: Any, _content: str, language: str
    ) -> list[dict[str, Any]]:
        """Extract logical blocks from a function AST node.

        This method identifies major code blocks within a function such as:
        - Loop bodies (for, while)
        - Conditional blocks (if/else)
        - Try/catch blocks
        - Large statement sequences

        Args:
            node: The function AST node
            content: The source code content
            language: Programming language

        Returns:
            List of dicts containing block info (type, start_pos, end_pos)
        """
        blocks = []

        # First, check if the function body itself is small enough
        if node.end_byte - node.start_byte <= self.config.max_function_chunk_size:
            return [
                {
                    "type": "whole_function",
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            ]

        # Extract the function body node
        body_node = None
        if language == "python":
            for child in node.children:
                if child.type == "block":
                    body_node = child
                    break
        elif language in ["javascript", "typescript"]:
            for child in node.children:
                if child.type == "statement_block":
                    body_node = child
                    break

        if not body_node:
            # No body found, return the whole function
            return [
                {
                    "type": "whole_function",
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            ]

        # Extract major blocks from the body
        current_block_start = body_node.start_byte
        accumulated_size = 0

        for child in body_node.children:
            child_size = child.end_byte - child.start_byte

            # Check if this child is a major block structure
            is_major_block = child.type in [
                # Python
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "with_statement",
                # JavaScript/TypeScript
                "if_statement",
                "for_statement",
                "while_statement",
                "do_statement",
                "try_statement",
                "switch_statement",
            ]

            # If adding this child would exceed our limit, or it's a major block
            if (
                accumulated_size + child_size > self.config.chunk_size // 2 and blocks
            ) or (is_major_block and accumulated_size > 0):
                # Save the current block
                blocks.append(
                    {
                        "type": "statement_sequence",
                        "start_pos": current_block_start,
                        "end_pos": child.start_byte,
                    }
                )
                current_block_start = child.start_byte
                accumulated_size = child_size
            else:
                accumulated_size += child_size

        # Add the final block
        if current_block_start < body_node.end_byte:
            blocks.append(
                {
                    "type": "statement_sequence",
                    "start_pos": current_block_start,
                    "end_pos": body_node.end_byte,
                }
            )

        # If we only got one block, just return the whole function
        if len(blocks) <= 1:
            return [
                {
                    "type": "whole_function",
                    "start_pos": node.start_byte,
                    "end_pos": node.end_byte,
                }
            ]

        # Include function signature with first block
        if blocks:
            blocks[0]["start_pos"] = node.start_byte

        return blocks

    def _extract_function_signature(
        self, node: Any, content: str, language: str
    ) -> str:
        """Extract the function signature from a function AST node.

        Args:
            node: The function AST node
            content: The source code content
            language: Programming language

        Returns:
            The function signature as a string
        """
        if language == "python":
            # Find the colon that ends the signature
            for child in node.children:
                if child.type == ":":
                    sig_end = child.start_byte
                    signature = content[node.start_byte : sig_end].strip()
                    return signature
        elif language in ["javascript", "typescript"]:
            # Find the opening brace of the function body
            for child in node.children:
                if child.type == "statement_block":
                    sig_end = child.start_byte
                    signature = content[node.start_byte : sig_end].strip()
                    return signature

        # Fallback: just get the first line
        first_line_end = content.find("\n", node.start_byte)
        if first_line_end > node.start_byte:
            return content[node.start_byte : first_line_end].strip()

        return ""

    def _get_js_method_name(self, node: Any, content: str) -> str:
        """Extract method name from JavaScript/TypeScript method node.

        Args:
            node: The method AST node
            content: The source code content

        Returns:
            The method name
        """
        for child in node.children:
            if child.type in ["property_identifier", "identifier"]:
                return content[child.start_byte : child.end_byte]
        return ""

    def _basic_chunking(self, content: str) -> list[Chunk]:
        """Basic character-based chunking (legacy)"""
        chunks = []
        pos = 0

        while pos < len(content):
            chunk_end = min(pos + self.config.chunk_size, len(content))

            # Basic boundary detection
            if chunk_end < len(content):
                # Look for sentence endings
                for boundary in [".\n", "\n\n", ". ", "!\n", "?\n"]:
                    boundary_idx = content.rfind(
                        boundary,
                        pos + self.config.chunk_size - 200,
                        chunk_end,
                    )
                    if boundary_idx > pos:
                        chunk_end = boundary_idx + len(boundary)
                        break

            chunk_content = content[pos:chunk_end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        start_pos=pos,
                        end_pos=chunk_end,
                        chunk_index=len(chunks),
                        chunk_type="text",
                    )
                )

            # Move with overlap
            pos = max(
                pos + self.config.chunk_size - self.config.chunk_overlap,
                chunk_end,
            )

        # Update metadata
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            chunk.char_count = len(chunk.content)
            chunk.token_estimate = chunk.char_count // 4

        return chunks

    def _format_chunks(
        self, chunks: list[Chunk], title: str, url: str
    ) -> list[dict[str, Any]]:
        """Format chunks into dictionary format expected by the system"""
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
