import sys
from unittest.mock import MagicMock

# Force VisionVQAModel to be None when tools.py imports it
sys.modules['specialist_models.vision_vqa'] = None
sys.modules['specialist_models'] = MagicMock()

import asyncio
import traceback
from ai_agent.agent_core.orchestrator import Orchestrator
from ai_agent.agent_core.state import AgentStateModel

def test():
    orchestrator = Orchestrator()
    query = 'Are there any open green spaces or water channels visible within this dense urban fabric?'
    model = AgentStateModel(
        raw_query=query,
        query=query,
        modalities={'uploaded_images': [{'file_path': 'dummy.jpg', 'modality': 'optical', 'file_name': 'dummy.jpg'}]}
    )
    state = model.to_graph_state()
    print('Running orchestrator with mocked None models...')
    try:
        result = orchestrator.run(query, state)
        import json
        json.dumps(result)  # TEST JSON SERIALIZATION
        print('SUCCESS:', result.get('status'))
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
