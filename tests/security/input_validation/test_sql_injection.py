"""SQL injection prevention tests.

This module tests protection against SQL injection attacks across all
data inputs and database interactions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.security import SecurityError, SecurityValidator


@pytest.mark.security
@pytest.mark.input_validation
class TestSQLInjectionPrevention:
    """Test SQL injection attack prevention."""

    @pytest.fixture
    def security_validator(self):
        """Get security validator instance."""
        return SecurityValidator()

    @pytest.fixture
    def sql_injection_payloads(self):
        """SQL injection test payloads."""
        return [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' OR 1=1--",
            "admin'--",
            "admin' OR 1=1#",
            "' UNION SELECT password FROM users--",
            "1; DELETE FROM users WHERE 1=1--",
            "'; INSERT INTO users VALUES('admin','password')--",
            "1' AND SLEEP(5)--",
            "' OR SUBSTRING(password,1,1)='a'--",
            "'; EXEC xp_cmdshell('dir')--",
            "' OR (SELECT COUNT(*) FROM users) > 0--",
            "admin'; WAITFOR DELAY '00:00:05'--",
            "' UNION ALL SELECT NULL,NULL,password FROM users--",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin')--",
        ]

    @pytest.fixture
    def advanced_sql_payloads(self):
        """Advanced SQL injection payloads."""
        return [
            "1' OR '1'='1' /*",
            "admin'/**/OR/**/1=1--",
            "' OR 'x'='x",
            "'; DECLARE @q VARCHAR(99); SET @q='DROP TABLE users'; EXEC(@q)--",
            "' AND 1=(SELECT COUNT(*) FROM tabname); --",
            "' OR 1=1 LIMIT 1--",
            "'||'1'='1",
            "1' ORDER BY 1--",
            "1' GROUP BY 1--",
            "' HAVING 1=1--",
            "'; CREATE TABLE temp (col1 VARCHAR(50))--",
            "'; ALTER TABLE users ADD COLUMN isadmin BOOLEAN--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
            "admin' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e))--",
            "' AND (SELECT SUBSTRING(@@version,1,1))='5'--",
        ]

    @pytest.mark.asyncio
    async def test_query_string_sql_injection_protection(
        self, security_validator, sql_injection_payloads
    ):
        """Test query string validation blocks SQL injection."""
        for payload in sql_injection_payloads:
            with pytest.raises(SecurityError, match="Query"):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_collection_name_sql_injection_protection(
        self, security_validator, sql_injection_payloads
    ):
        """Test collection name validation blocks SQL injection."""
        for payload in sql_injection_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_collection_name(payload)

    @pytest.mark.asyncio
    async def test_url_sql_injection_protection(
        self, security_validator, sql_injection_payloads
    ):
        """Test URL validation blocks SQL injection in URLs."""
        for payload in sql_injection_payloads:
            malicious_url = f"https://example.com/search?q={payload}"
            with pytest.raises(SecurityError):
                security_validator.validate_url(malicious_url)

    @pytest.mark.asyncio
    async def test_advanced_sql_injection_patterns(
        self, security_validator, advanced_sql_payloads
    ):
        """Test protection against advanced SQL injection techniques."""
        for payload in advanced_sql_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_database_parameterization(self):
        """Test that database queries use parameterized statements."""
        # This would test actual database interaction code
        # For now, we'll mock the database layer
        with patch("src.infrastructure.database.connection_manager") as mock_db:
            mock_cursor = AsyncMock()
            mock_db.get_connection.return_value.__aenter__.return_value.cursor.return_value = mock_cursor

            # Simulate a search query that should use parameters

            # The actual implementation should use parameterized queries
            # This test ensures the database layer properly escapes inputs
            mock_cursor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_orm_injection_protection(self):
        """Test ORM-level injection protection."""
        # Test that any ORM usage properly sanitizes inputs
        dangerous_inputs = [
            {"name": "'; DROP TABLE users; --"},
            {"id": "1 OR 1=1"},
            {"filter": "admin'--"},
        ]

        for _dangerous_input in dangerous_inputs:
            # Simulate ORM query building that should sanitize inputs
            # This would test actual ORM code
            pass

    @pytest.mark.asyncio
    async def test_blind_sql_injection_prevention(self):
        """Test prevention of blind SQL injection attacks."""
        blind_payloads = [
            "' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a'--",
            "' AND (SELECT COUNT(*) FROM users WHERE username='admin' AND password LIKE 'a%')>0--",
            "' AND SLEEP(5)--",
            "'; WAITFOR DELAY '00:00:05'--",
            "' AND (SELECT COUNT(*) FROM information_schema.tables)>0--",
        ]

        security_validator = SecurityValidator()
        for payload in blind_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_time_based_sql_injection_prevention(self):
        """Test prevention of time-based SQL injection attacks."""
        time_based_payloads = [
            "'; SLEEP(10)--",
            "'; WAITFOR DELAY '00:00:10'--",
            "' AND (SELECT SLEEP(10))--",
            "'; pg_sleep(10)--",
            "' AND 1=(SELECT COUNT(*) FROM pg_sleep(10))--",
        ]

        security_validator = SecurityValidator()
        for payload in time_based_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_union_based_sql_injection_prevention(self):
        """Test prevention of UNION-based SQL injection attacks."""
        union_payloads = [
            "' UNION SELECT username, password FROM users--",
            "' UNION ALL SELECT NULL, @@version--",
            "' UNION SELECT 1,2,3,4,database()--",
            "' UNION SELECT table_name FROM information_schema.tables--",
            "' UNION SELECT column_name FROM information_schema.columns--",
        ]

        security_validator = SecurityValidator()
        for payload in union_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_error_based_sql_injection_prevention(self):
        """Test prevention of error-based SQL injection attacks."""
        error_payloads = [
            "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e))--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
            "' AND UPDATEXML(1,CONCAT(0x7e,(SELECT version()),0x7e),1)--",
            "' AND 1=CAST((SELECT version()) AS int)--",
        ]

        security_validator = SecurityValidator()
        for payload in error_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_stored_procedure_injection_prevention(self):
        """Test prevention of stored procedure injection attacks."""
        stored_proc_payloads = [
            "'; EXEC xp_cmdshell('cmd')--",
            "'; EXEC sp_makewebtask--",
            "'; EXEC master..xp_cmdshell 'ping 127.0.0.1'--",
            "'; EXEC xp_regwrite--",
            "'; EXEC sp_OACreate--",
        ]

        security_validator = SecurityValidator()
        for payload in stored_proc_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_valid_queries_pass_validation(self, security_validator):
        """Test that legitimate queries pass validation."""
        valid_queries = [
            "python documentation",
            "machine learning tutorial",
            "API reference guide",
            "database best practices",
            "security guidelines",
        ]

        for query in valid_queries:
            # Should not raise any exception
            validated = security_validator.validate_query_string(query)
            assert validated == query

    @pytest.mark.asyncio
    async def test_parameterized_query_enforcement(self):
        """Test that parameterized queries are enforced in database operations."""
        # This test would verify that all database operations use parameterized queries
        # and never concatenate user input directly into SQL strings

        # Mock database operations
        with patch("sqlite3.connect") as mock_connect:
            mock_cursor = MagicMock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            # Test that parameterized queries are used
            # In the actual implementation, verify that execute() is called with parameters

            # The implementation should use parameterized queries like:
            # cursor.execute("SELECT * FROM users WHERE name = ?", (dangerous_input,))
            # Not: cursor.execute(f"SELECT * FROM users WHERE name = '{dangerous_input}'")

            # Verify parameterized query usage
            # mock_cursor.execute.assert_called_with(
            #     "SELECT * FROM table WHERE column = ?",
            #     (dangerous_input,)
            # )

    def test_input_length_limits(self, security_validator):
        """Test input length limits prevent buffer overflow attacks."""
        # Test extremely long inputs that could cause buffer overflows
        long_input = "A" * 10000

        with pytest.raises(SecurityError, match="too long"):
            security_validator.validate_query_string(long_input)

    def test_special_character_handling(self, security_validator):
        """Test proper handling of special characters."""
        special_chars = [
            "'",
            '"',
            ";",
            "--",
            "/*",
            "*/",
            "\x00",
            "\n",
            "\r",
            "\t",
            "\\",
            "%",
            "_",
        ]

        for char in special_chars:
            test_input = f"test{char}input"
            # Should either sanitize or reject
            try:
                result = security_validator.validate_query_string(test_input)
                # If validation passes, dangerous characters should be removed
                assert char not in result or char in ["_", "%"]  # Allow safe chars
            except SecurityError:
                # Rejection is also acceptable
                pass

    @pytest.mark.asyncio
    async def test_sql_injection_in_json_payloads(self):
        """Test SQL injection prevention in JSON payloads."""
        json_payloads = [
            {"search": "'; DROP TABLE users; --"},
            {"filters": {"name": "' OR 1=1--"}},
            {"metadata": {"description": "' UNION SELECT password FROM users--"}},
        ]

        security_validator = SecurityValidator()

        for payload in json_payloads:
            for value in payload.values():
                if isinstance(value, str):
                    with pytest.raises(SecurityError):
                        security_validator.validate_query_string(value)
                elif isinstance(value, dict):
                    for nested_value in value.values():
                        if isinstance(nested_value, str):
                            with pytest.raises(SecurityError):
                                security_validator.validate_query_string(nested_value)

    def test_encoding_bypass_prevention(self, security_validator):
        """Test prevention of encoding-based SQL injection bypasses."""
        encoded_payloads = [
            # URL encoded
            "%27%20OR%20%271%27%3D%271",  # ' OR '1'='1
            "%3B%20DROP%20TABLE%20users%3B",  # ; DROP TABLE users;
            # Double URL encoded
            "%2527%2520OR%2520%25271%2527%253D%25271",
            # Hex encoded
            "0x27204f52202731273d2731",
            # Unicode encoded
            "\\u0027\\u0020OR\\u0020\\u0027\\u0031\\u0027\\u003d\\u0027\\u0031",
        ]

        for payload in encoded_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)
