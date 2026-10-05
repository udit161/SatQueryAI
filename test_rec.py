"""
Test to reproduce the maximum recursion depth exceeded error at RUNTIME
(not at import time) when processing a VQA query through the orchestrator.
"""
import sys
sys.setrecursionlimit(500)  # keep it LOW to surface real recursion

try:
    from ai_agent.agent_core.orchestrator import Orchestrator
    from ai_agent.agent_core.state import AgentStateModel

    model = AgentStateModel(
        raw_query="Are there any open green spaces or water channels visible within this dense urban fabric?",
        query="Are there any open green spaces or water channels visible within this dense urban fabric?",
    )
    initial_state = model.to_graph_state()
    print("to_graph_state OK")

    orch = Orchestrator()
    print("Orchestrator init OK")

    result = orch.run("Are there any open green spaces or water channels visible within this dense urban fabric?", initial_state)
    print("run OK:", list(result.keys()) if isinstance(result, dict) else type(result))

except RecursionError:
    import traceback
    traceback.print_exc()
except Exception as e:
    import traceback
    print('Error:', type(e).__name__, str(e)[:400])
    traceback.print_exc()
