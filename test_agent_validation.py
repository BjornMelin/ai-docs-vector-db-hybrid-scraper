#!/usr/bin/env python3
"""
Focused Agent Validation Script

Tests core agent functionality with fallback modes to ensure production readiness.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_agent_core_functionality():
    """Test core agent functionality and fallback modes."""
    print("🧪 Testing Agent Core Functionality")
    print("=" * 50)
    
    # Test 1: Agent State
    try:
        from src.services.agents.core import AgentState
        state = AgentState(session_id="test-session")
        state.add_interaction("user", "test message")
        state.update_metrics({"test_metric": 1.0})
        state.increment_tool_usage("test_tool")
        print("✅ AgentState: OK")
    except Exception as e:
        print(f"❌ AgentState: {e}")
        return False

    # Test 2: Agent Dependencies
    try:
        from src.services.agents.core import BaseAgentDependencies, create_agent_dependencies
        from src.config.settings import Settings
        
        class MockClientManager:
            pass
        
        mock_client = MockClientManager()
        deps = create_agent_dependencies(mock_client, session_id="test")
        print("✅ Agent Dependencies: OK")
    except Exception as e:
        print(f"❌ Agent Dependencies: {e}")
        return False

    # Test 3: Agent Fallback Execution
    try:
        from src.services.agents.core import BaseAgent, AgentState, BaseAgentDependencies
        from src.config.settings import Settings
        
        # Create test agent
        class TestAgent(BaseAgent):
            def get_system_prompt(self) -> str:
                return "Test agent for validation"
            
            async def initialize_tools(self, deps) -> None:
                pass
        
        agent = TestAgent("test-agent")
        
        # Create mock dependencies
        class MockClientManager:
            pass
        
        deps = BaseAgentDependencies(
            client_manager=MockClientManager(),
            config=Settings(),
            session_state=AgentState(session_id="test")
        )
        
        # Test execution (should use fallback mode without API keys)
        result = await agent.execute("test search task", deps)
        
        if result and result.get("success"):
            print("✅ Agent Fallback Execution: OK")
            print(f"   Fallback used: {result.get('fallback_used', False)}")
            print(f"   Fallback reason: {result.get('fallback_reason', 'N/A')}")
        else:
            print(f"❌ Agent Fallback Execution: Failed - {result}")
            return False
            
    except Exception as e:
        print(f"❌ Agent Fallback Execution: {e}")
        return False

    # Test 4: Agent Registry
    try:
        from src.services.agents.core import AgentRegistry
        
        registry = AgentRegistry()
        registry.register_agent(agent)
        
        retrieved_agent = registry.get_agent("test-agent")
        if retrieved_agent and retrieved_agent.name == "test-agent":
            print("✅ Agent Registry: OK")
        else:
            print("❌ Agent Registry: Failed")
            return False
    except Exception as e:
        print(f"❌ Agent Registry: {e}")
        return False

    return True


async def test_agentic_orchestrator():
    """Test agentic orchestrator functionality."""
    print("\n🎭 Testing Agentic Orchestrator")
    print("=" * 50)
    
    try:
        from src.services.agents.agentic_orchestrator import AgenticOrchestrator, ToolRequest
        from src.services.agents.core import BaseAgentDependencies, AgentState
        from src.config.settings import Settings
        
        class MockClientManager:
            pass
        
        # Create orchestrator
        orchestrator = AgenticOrchestrator()
        
        # Create dependencies
        deps = BaseAgentDependencies(
            client_manager=MockClientManager(),
            config=Settings(),
            session_state=AgentState(session_id="test")
        )
        
        # Test orchestration (should use fallback mode)
        request = ToolRequest(task="test orchestration task")
        response = await orchestrator.orchestrate(request, deps)
        
        if response and response.success:
            print("✅ Agentic Orchestrator: OK")
            print(f"   Success: {response.success}")
            print(f"   Results keys: {list(response.results.keys()) if response.results else 'None'}")
        else:
            print(f"❌ Agentic Orchestrator: Failed - {response}")
            return False
            
    except Exception as e:
        print(f"❌ Agentic Orchestrator: {e}")
        return False
    
    return True


async def test_configuration_system():
    """Test unified configuration system."""
    print("\n⚙️ Testing Configuration System")
    print("=" * 50)
    
    try:
        from src.config import get_config
        from src.config.settings import Settings
        
        # Test config loading
        config = get_config()
        if config:
            print("✅ Config Loading: OK")
        else:
            print("❌ Config Loading: Failed")
            return False
        
        # Test settings
        settings = Settings()
        if hasattr(settings, 'openai') and hasattr(settings, 'qdrant'):
            print("✅ Settings Structure: OK")
        else:
            print("❌ Settings Structure: Missing required sections")
            return False
            
        # Test environment variable support
        original_env = os.environ.get("AI_DOCS_APP_NAME")
        os.environ["AI_DOCS_APP_NAME"] = "Test App"
        test_settings = Settings()
        if test_settings.app_name == "Test App":
            print("✅ Environment Variables: OK")
        else:
            print("❌ Environment Variables: Failed")
        
        # Restore original env
        if original_env:
            os.environ["AI_DOCS_APP_NAME"] = original_env
        else:
            os.environ.pop("AI_DOCS_APP_NAME", None)
            
    except Exception as e:
        print(f"❌ Configuration System: {e}")
        return False
    
    return True


async def test_mcp_services():
    """Test MCP services basic functionality."""
    print("\n🔧 Testing MCP Services")
    print("=" * 50)
    
    try:
        # Test service imports
        from src.mcp_services.analytics_service import AnalyticsService
        from src.mcp_services.document_service import DocumentService
        from src.mcp_services.search_service import SearchService
        
        # Test service instantiation
        analytics = AnalyticsService()
        document = DocumentService()
        search = SearchService()
        
        print("✅ MCP Service Imports: OK")
        print("✅ MCP Service Instantiation: OK")
        
    except Exception as e:
        print(f"❌ MCP Services: {e}")
        return False
    
    return True


async def main():
    """Main validation function."""
    print("🚀 Production Agent Validation Suite")
    print("=" * 60)
    
    # Check API key availability
    api_keys_available = bool(
        os.getenv("OPENAI_API_KEY") and 
        os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    )
    print(f"🔑 API Keys Available: {api_keys_available}")
    
    # Check Pydantic-AI availability
    try:
        import pydantic_ai
        pydantic_ai_available = True
    except ImportError:
        pydantic_ai_available = False
    print(f"🤖 Pydantic-AI Available: {pydantic_ai_available}")
    
    print()
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Agent Core Functionality", test_agent_core_functionality),
        ("Agentic Orchestrator", test_agentic_orchestrator),
        ("MCP Services", test_mcp_services),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    success_rate = (passed / total) * 100
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 85.0:
        print("✅ VALIDATION PASSED - Production ready with fallback modes")
    else:
        print("❌ VALIDATION FAILED - Needs attention")
    
    return success_rate >= 85.0


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)